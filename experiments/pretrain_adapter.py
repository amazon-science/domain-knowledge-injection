import os
import sys
import json
import torch
import random
import logging
import argparse

import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartTokenizer

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from adapter_model import PretrainedModel, AdapterModel
from util import PROCESSORS, convert_examples_to_features_ki, _rotate_checkpoints

from nltk.translate.bleu_score import sentence_bleu


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
logger = logging.getLogger(__name__)


# Can be optionally extended to support other pre-trained models
MODEL_CLASSES = {
    'bart': (BartForConditionalGeneration, BartTokenizer),
}


def set_seed(args):
    """ Set seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, models, tokenizer, current_domain):
    """ Train the adapter for the current_domain """

    # Set-up TensorBoard logger
    tb_writer = None
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # Designate models
    pretrained_model = models[0]
    adapter_model = models[1]
    main_model = pretrained_model if adapter_model is None else adapter_model

    # Set batch size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Handle data serving
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # Determine total number of training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Set-up weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in main_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in main_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Set-up optimizer and schedule (linear warmup)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = \
        get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Set-up mixed precision training
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')
        model, optimizer = amp.initialize(main_model, optimizer, opt_level=args.fp16_opt_level)

    # Set-up multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        if adapter_model is not None:
            adapter_model = torch.nn.DataParallel(adapter_model)

    # Set-up distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        pretrained_model = torch.nn.parallel.DistributedDataParallel(pretrained_model,
                                                                     device_ids=[args.local_rank],
                                                                     output_device=args.local_rank)
        if adapter_model is not None:
            adapter_model = torch.nn.parallel.DistributedDataParallel(adapter_model,
                                                                      device_ids=[args.local_rank],
                                                                      output_device=args.local_rank)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = {}".format(len(train_dataset)))
    logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
                args.train_batch_size * args.gradient_accumulation_steps * \
                (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(t_total))

    # Initialize trackers and zero model gradients
    global_step = 0
    stale_epochs = 0
    best_memo_acc = 0.
    mean_epoch_loss, memo_acc = None, None
    pretrained_model.zero_grad()
    if adapter_model is not None:
        adapter_model.zero_grad()

    # Iterate over update steps
    best_checkpoint_path = None
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
    for epoch_id, _ in enumerate(train_iterator):

        epoch_losses = []
        epoch_iterator = \
            tqdm(train_dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0], mininterval=10, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            main_model.train()
            if adapter_model is not None:
                pretrained_model.freeze_plm_params()  # Freeze parameters of the pre-trained model

            # Perform a single forward pass
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            # Prepare decoder inputs and labels for enc-dec models
            inputs['labels'] = batch[3][:, 1:].clone().contiguous()  # shift
            decoder_input_ids = batch[3][:, :-1].clone().contiguous()  # shift
            decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id  # remove masking
            inputs['decoder_input_ids'] = decoder_input_ids.contiguous()
            inputs['decoder_attention_mask'] = batch[4][:, :-1].clone().contiguous()

            if adapter_model is not None:
                pretrained_model_outputs = pretrained_model(**inputs, return_dict=True)
                outputs = adapter_model(pretrained_model_outputs, current_domain, **inputs, return_dict=False)
            else:
                outputs = main_model(**inputs, return_dict=False)

            # Loss is obtained from the full model
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            epoch_losses.append(loss.item())

            # Compute distributed loss
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Compute mixed-precision loss
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Back-propagate
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                pretrained_model.zero_grad()
                if adapter_model is not None:
                    adapter_model.zero_grad()
                global_step += 1

            if global_step > args.max_steps > 0:
                epoch_iterator.close()
                break

        # Report losses
        mean_epoch_loss = np.mean(epoch_losses)
        logging.info('\n' + '*' * 10)
        logging.info('Mean epoch training loss: {:.4f}'.format(mean_epoch_loss))
        logging.info('Current learning rate: {}'.format(scheduler.get_lr()[0]))
        logging.info('*' * 10 + '\n')

        # Track training
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('train_loss', mean_epoch_loss, global_step)

        # Periodically report stats and check whether to stop fine-tuning
        if epoch_id > 0 and epoch_id % args.eval_epochs == 0:

            if args.local_rank in [-1, 0]:
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.local_rank == -1 and args.evaluate_during_training:
                    results, memo_acc, bleu = \
                        evaluate(args, models, tokenizer, current_domain, epoch=epoch_id, step=global_step)

            # Save model checkpoint
            if args.local_rank in [-1, 0]:
                checkpoint_path = os.path.join(args.output_dir, 'checkpoint-{}_adapter-epoch_{}-gs_{}'.format(
                    '_'.join(args.active_domains), epoch_id, global_step))
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                # Take care of distributed / parallel training
                if adapter_model is not None:
                    model_to_save = adapter_model.module if hasattr(adapter_model, 'module') else adapter_model
                else:
                    model_to_save = pretrained_model.model.module if hasattr(pretrained_model.model, 'module') else \
                        pretrained_model.model
                model_to_save.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                torch.save(
                    args, os.path.join(checkpoint_path, 'training_args_{}.bin'.format('_'.join(args.active_domains))))
                logger.info('Saving model checkpoint to {}'.format(checkpoint_path))
                # Delete old checkpoints to maintain a low memory footprint
                if args.save_total_limit > 0:
                    _rotate_checkpoints(
                        args, 'checkpoint-{}_adapter'.format('_'.join(args.active_domains)), use_mtime=False)

            # Check whether to stop training early
            if best_memo_acc < memo_acc:
                best_memo_acc = memo_acc
                stale_epochs = 0
                best_checkpoint_path = \
                    os.path.join(args.output_dir, 'checkpoint-{}_adapter-best'.format('_'.join(args.active_domains)))
                # Take care of distributed / parallel training
                if adapter_model is not None:
                    model_to_save = adapter_model.module if hasattr(adapter_model, 'module') else adapter_model
                else:
                    model_to_save = pretrained_model.model.module if hasattr(pretrained_model.model, 'module') else \
                        pretrained_model.model
                model_to_save.save_pretrained(best_checkpoint_path)
                tokenizer.save_pretrained(best_checkpoint_path)
                torch.save(args, os.path.join(best_checkpoint_path,
                                              'training_args_{}.bin'.format('_'.join(args.active_domains))))
                logger.info('Saving model checkpoint to {}'.format(best_checkpoint_path))

            else:
                stale_epochs += 1
                logging.info(
                    '!!! Memorization has not improved this epoch. Stale epochs: {} !!!'.format(stale_epochs))

            if stale_epochs >= args.patience:
                logging.info(
                    '\n***** STOPPING TRAINING EARLY AFTER {} STALE TRAINING EPOCHS *****\n'.format(stale_epochs))
                break

            if global_step > args.max_steps > 0:
                train_iterator.close()
                break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, mean_epoch_loss, best_checkpoint_path


def evaluate(args, models, tokenizer, domain, epoch=0, step=0, show_low_bleu_seqs=False, no_logging=False):
    """ Checks the success of knowledge injection by letting the model generate masked DB information """

    pretrained_model = models[0]
    adapter_model = models[1]
    main_model = pretrained_model.model if adapter_model is None else adapter_model

    results = dict()
    if not no_logging:
        # Reference previous evaluation results
        if os.path.exists(os.path.join(args.output_dir, 'test_metrics.json')):
            with open(os.path.join(args.output_dir, 'test_metrics.json'), 'r') as f:
                existing_results = json.loads(f.read())
            f.close()
            results.update(existing_results)

    # Set up data-serving pipeline
    test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset[domain][0]) if args.local_rank == -1 \
        else DistributedSampler(test_dataset[domain][0])
    test_dataloader = DataLoader(test_dataset[domain][0], sampler=test_sampler, batch_size=test_batch_size)
    # Unpack
    all_slots, all_slot_seq, all_entities = test_dataset[domain][1], test_dataset[domain][2], test_dataset[domain][3]

    # Test!
    logger.info('***** Testing generation on the training set *****')
    logger.info('  Num examples = %d', len(test_dataset[domain][0]))
    logger.info('  Batch size = %d', test_batch_size)
    generation_inputs = list()
    generation_targets = list()
    generated_sequences = list()
    slot_success_rates = dict()

    # Iterate through the test corpus
    main_model.eval()
    for batch_id, batch in enumerate(tqdm(test_dataloader, desc='Testing', mininterval=10, ncols=100)):
        batch = tuple(t.to(args.device) for t in batch)

        input_ids = batch[0]
        attention_mask = batch[1]
        target_ids = batch[3]
        gen_prompt = tokenizer.eos_token_id

        max_gen_length = args.max_gen_length
        outputs = main_model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      min_length=5,
                                      max_length=max_gen_length,
                                      temperature=args.temperature,
                                      top_k=args.k if args.k > 0 else None,
                                      top_p=args.p if args.p > 0 else None,
                                      num_beams=1,  # greedy decoding
                                      do_sample=args.do_sample,
                                      early_stopping=True,
                                      no_repeat_ngram_size=3,
                                      decoder_start_token_id=gen_prompt)

        # Remove the batch dimension when returning multiple sequences
        if len(outputs.shape) > 2:
            outputs.squeeze_()

        # Post-process model predictions and prediction targets
        for generated_sequence_id, generated_sequence in enumerate(outputs):
            generated_sequence = generated_sequence.tolist()[1:]

            # Prepare predictions
            try:
                generated_sequence = generated_sequence[: generated_sequence.index(tokenizer.eos_token_id)]
            except ValueError:
                pass
            decoded_generated_sequence = \
                tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=False)
            generated_sequences.append(decoded_generated_sequence)

            # Prepare inputs
            gen_input = input_ids[generated_sequence_id].tolist()[1:]
            try:
                gen_input = gen_input[: gen_input.index(tokenizer.eos_token_id)]
            except ValueError:
                pass
            generation_inputs.append(tokenizer.decode(gen_input, clean_up_tokenization_spaces=True))

            # Prepare generation targets
            gen_target = target_ids[generated_sequence_id].tolist()[1:]
            try:
                gen_target = gen_target[: gen_target.index(tokenizer.eos_token_id)]
            except ValueError:
                pass
            decoded_generation_target = tokenizer.decode(gen_target, clean_up_tokenization_spaces=True)
            generation_targets.append(decoded_generation_target)

    assert len(generation_inputs) == len(generation_targets) == len(generated_sequences) == len(all_slots), \
        'Generation output length mismatch! {}, {}, {}, {}'.format(len(generation_inputs), len(generation_targets),
                                                                   len(generated_sequences), len(all_slots))

    # Report sample generation results
    logging.info('***** Example generations *****')
    for s_id, gen_input in enumerate(generation_inputs):
        if s_id >= 10:
            break
        logging.info('-' * 10)
        logging.info('  Input: {:s}'.format(gen_input))
        logging.info('  Slot: {:s}'.format(all_slots[s_id]))
        logging.info('  Slot in sequence: {:s}'.format(str(all_slot_seq[s_id])))
        logging.info('  Entity: {:s}'.format(all_entities[s_id]))
        logging.info('  Target: {:s}'.format(generation_targets[s_id]))
        logging.info('  Prediction: {:s}'.format(generated_sequences[s_id]))

    # Compute and update evaluation metric values
    target_present, target_missing = 0, 0
    bleu_scores = []
    hits_per_domain = {domain: [0, 0] for domain in ['restaurant', 'hotel', 'attraction', 'train']}

    for seq_id, seq in enumerate(generated_sequences):
        if 'restaurant' in generation_targets[seq_id]:
            fact_domain = 'restaurant'
        elif 'hotel' in generation_targets[seq_id]:
            fact_domain = 'hotel'
        elif 'attraction' in generation_targets[seq_id]:
            fact_domain = 'attraction'
        else:
            fact_domain = 'train'

        target_success = False
        # Track slots
        slot = all_slots[seq_id]
        if slot_success_rates.get(slot, None) is None:
            slot_success_rates[slot] = {'present': 0, 'missing': 0}

        # Check if DB information was retrieved (also checks for the correctness of the entity and the relation)
        # Similar to BLEU, but focused on DB-relevant information
        if generation_targets[seq_id].lower().strip() in seq.lower().strip() and \
                all_entities[seq_id].lower().strip() in seq.lower().strip() and \
                (all_slot_seq[seq_id] is None or (all_slot_seq[seq_id] is not None and
                                                  all_slot_seq[seq_id].lower().strip() in seq.lower().strip())):
            target_present += 1
            if args.active_domains[0] == 'mixed':
                hits_per_domain[fact_domain][0] += 1
            slot_success_rates[slot]['present'] += 1
            target_success = True
        else:
            target_missing += 1
            if args.active_domains[0] == 'mixed':
                hits_per_domain[fact_domain][1] += 1
            slot_success_rates[slot]['missing'] += 1

        # Compute sentence-level BLEU
        reference = generation_inputs[seq_id].replace(tokenizer.mask_token, generation_targets[seq_id])
        sent_bleu = sentence_bleu([reference.strip().split(' ')], seq.strip().split(' '), weights=[1])
        bleu_scores.append(sent_bleu)

        if show_low_bleu_seqs and (target_success and sent_bleu < 1.0):
            logging.info('-' * 10)
            logging.info('REFERENCE: {}'.format(reference))
            logging.info('GENERATION: {}'.format(seq))
            logging.info('BLEU: {}'.format(sent_bleu))

    db_memorization_accuracy = target_present / (target_present + target_missing)
    domain_memorization_accuracies = {domain: 0. for domain in ['restaurant', 'hotel', 'attraction', 'train']}
    if args.active_domains[0] == 'mixed':
        for domain in hits_per_domain.keys():
            if sum(hits_per_domain[domain]) > 0:
                domain_memorization_accuracies[domain] = hits_per_domain[domain][0] / sum(hits_per_domain[domain])
    mean_bleu = np.mean(bleu_scores)

   # Normalize slot success rates
    norm_slot_success_rates = {slot: {'present': 0, 'missing': 0} for slot in slot_success_rates.keys()}
    for slot in slot_success_rates.keys():
        norm_slot_success_rates[slot]['present'] = \
            (slot_success_rates[slot]['present'], slot_success_rates[slot]['present'] /
             (slot_success_rates[slot]['present'] + slot_success_rates[slot]['missing']))
        norm_slot_success_rates[slot]['missing'] = \
            (slot_success_rates[slot]['missing'], slot_success_rates[slot]['missing'] /
             (slot_success_rates[slot]['present'] + slot_success_rates[slot]['missing']))
    slot_success_rates_string = str(norm_slot_success_rates)

    if not no_logging:
        # Update results
        if results.get('epoch_and_step', None) is None:
            results['epoch_and_step'] = []
        if results.get('memorization_accuracy', None) is None:
            results['memorization_accuracy'] = []
        if results.get('mean_bleu', None) is None:
            results['mean_bleu'] = []
        if results.get('slot_success_rates', None) is None:
            results['slot_success_rates'] = []

        results['epoch_and_step'].append((epoch, step))
        results['memorization_accuracy'].append(db_memorization_accuracy)
        results['mean_bleu'].append(mean_bleu)
        results['slot_success_rates'].append(slot_success_rates_string)
        if args.active_domains[0] == 'mixed':
            if results.get('domain_memorization_accuracies', None) is None:
                results['domain_memorization_accuracies'] = []
            results['domain_memorization_accuracies'].append(domain_memorization_accuracies)

        # Log metrics
        output_eval_file = os.path.join(args.output_dir,
                                        'generation_test_results-{}_adapter-epoch_{}-gs_{}'.format(domain, epoch, step))
        with open(output_eval_file, 'w') as writer:
            writer.write('STEP: {:s}\n'.format(str(step)))
            writer.write('%s = %s\n' % ('memorization accuracy', str(db_memorization_accuracy)))
            writer.write('%s = %s\n' % ('mean sentence-BLEU', str(mean_bleu)))

    logger.info('***** Evaluation results *****')
    logger.info('  %s = %s', 'memorization accuracy', str(db_memorization_accuracy))
    logger.info('  %s = %s', 'mean sentence-BLEU', str(mean_bleu))
    logger.info('  %s = %s', 'slot success rates', slot_success_rates_string)
    if args.active_domains[0] == 'mixed':
        logger.info('Domain memorization accuracies:')
        for domain in domain_memorization_accuracies.keys():
            logger.info('{}: {}'.format(domain, domain_memorization_accuracies[domain]))

    # Log predictions
    output_pred_file = os.path.join(args.output_dir,
                                    'generation_test_predictions-{}_adapter-epoch_{}-gs_{}'.format(domain, epoch, step))
    with open(output_pred_file, 'w') as writer:
        logger.info('***** Write predictions *****')
        for gsi, gs in enumerate(generated_sequences):
            writer.write(json.dumps({'input': generation_inputs[gsi],
                                     'target': generation_targets[gsi],
                                     'prediction': gs}) + '\n')

    if not no_logging:
        # Maintain a single metrics file
        with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
            f.write(json.dumps(results))

    return results, db_memorization_accuracy, mean_bleu


# ======================================================================================================================
# Data preprocessing

def load_and_cache_examples(args, tokenizer, evaluate=False):
    """ Pre-process raw inputs for training the model """

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Select the right processor
    processor = PROCESSORS[args.fact_format](tokenizer)
    # Load data features from cache or dataset file
    cached_features_file = \
        os.path.join(args.data_dir, 'cached_{}_{}_{}_{}.bin'.format('_'.join(args.active_domains),
                                                                    args.fact_format, str(args.max_seq_length),
                                                                    'eval' if evaluate else ''))
    if os.path.exists(cached_features_file):
        # Load existing features
        logger.info('Loading features from cached file {}'.format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        # Generate sample features from raw input
        logger.info('Creating features from dataset file at {}'.format(args.data_dir))
        # Load samples
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        # Generate features
        features = convert_examples_to_features_ki(examples, args.max_seq_length, tokenizer)  # returns a dictionary
        # Save features for future retrieval
        if args.local_rank in [-1, 0]:
            logger.info('Saving features into cached file {}'.format(cached_features_file))
            torch.save(features, cached_features_file)

    # Convert features to tensors and build dataset
    all_datasets = dict()
    for domain in features.keys():
        if args.active_domains[0] == 'mixed':
            all_input_ids, all_input_mask, all_segment_ids, all_target_ids, all_target_mask = \
                list(), list(), list(), list(), list()
            all_slots, all_slot_seq, all_entities = list(), list(), list()

            # Collect
            for domain in features.keys():
                all_input_ids += [f.input_ids for f in features[domain]]
                all_input_mask += [f.input_mask for f in features[domain]]
                all_segment_ids += [f.segment_ids for f in features[domain]]
                all_target_ids += [f.target_ids for f in features[domain]]
                all_target_mask += [f.target_mask for f in features[domain]]
                all_slots += [f.slot for f in features[domain]]
                all_slot_seq += [f.slot_seq for f in features[domain]]
                all_entities += [f.entity for f in features[domain]]

            # Shuffle
            zipped = list(zip(all_input_ids, all_input_mask, all_segment_ids, all_target_ids, all_target_mask,
                              all_slots, all_slot_seq, all_entities))
            random.shuffle(zipped)
            all_input_ids, all_input_mask, all_segment_ids, all_target_ids, all_target_mask, all_slots, all_slot_seq, \
                all_entities = zip(*zipped)

            # Make tensors, if
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
            all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
            all_target_ids = torch.tensor(all_target_ids, dtype=torch.long)
            all_target_mask = torch.tensor(all_target_mask, dtype=torch.long)

            all_datasets['mixed'] = \
                (TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_target_ids, all_target_mask),
                 all_slots, all_slot_seq, all_entities)

        else:
            if domain in args.active_domains:
                all_input_ids = torch.tensor([f.input_ids for f in features[domain]], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features[domain]], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features[domain]], dtype=torch.long)
                all_target_ids = torch.tensor([f.target_ids for f in features[domain]], dtype=torch.long)
                all_target_mask = torch.tensor([f.target_mask for f in features[domain]], dtype=torch.long)
                all_slots = [f.slot for f in features[domain]]
                all_slot_seq = [f.slot_seq for f in features[domain]]
                all_entities = [f.entity for f in features[domain]]

                all_datasets[domain] = \
                    (TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_target_ids, all_target_mask),
                     all_slots, all_slot_seq, all_entities)
    return all_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='roberta', type=str, required=True,
                        help="Model type for the pre-trained model")
    parser.add_argument("--model_name_or_path", default='bart-large', type=str, required=True,
                        help="Path to relevant pre-trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="path to the output directory")

    # Adapter (training)-specific arguments
    parser.add_argument("--active_domains", type=str, nargs='+', required=True,
                        help="Names of the MultiWOZ domains used to train the current adapter")
    parser.add_argument("--fact_format", type=str, choices=['relational_tuples', 'atomic_facts', 'composite_facts',
                                                            'atomic_and_composite_facts'],
                        required=True, help="Format of the DB facts presented to the model")
    parser.add_argument("--adapter_size", type=int, default=192,  # 768 in K-Adapter
                        help="Dimensionality of the adapter attention layers")
    parser.add_argument("--adapter_num_heads", type=int, default=12,
                        help="Number of adapter attention heads")
    parser.add_argument("--adapter_num_layers", default=2, type=int,
                        help="Number of transformer layers in each adapter module")
    parser.add_argument("--adapter_list", default=None, type=str,
                        help="The pretrained language model layers after which adapters are added;"
                             "e.g.enc-1;enc-4;enc-12;dec-1;dec-4;dec-12")
    parser.add_argument("--adapter_skip_layers", default=0, type=int,
                        help="How many adapter layers to skip for skip connections between adapter layers")
    parser.add_argument("--initialize_adapters_from_layers", action='store_true',
                        help="Whether to initialize adapter layers with parameters of the pretrained LM layers.")
    parser.add_argument("--adapter_model_checkpoint", default=None,
                        help="Path to the checkpoint of the pre-trained adapter-model; used for resuming training and "
                             "isolated evaluation")
    parser.add_argument("--adapter_combo_method", type=str, default=None,
                        help="Format of the DB facts presented to the model")
    parser.add_argument("--plm_only", action='store_true',
                        help="Whether to fine-tune the pre=trained LM only, i.e. without taking adapters into account")
    parser.add_argument("--task", type=str, choices=['response_selection', 'response_generation', 'state_tracking',
                                                     'composite_facts_eval'],
                        default=None, help="Downstream task to fine-tune the model on")
    parser.add_argument("--no_encoder_integration", action='store_true',
                        help="Disables adapter integration into the encoder output.")


    ## Generation parameters
    parser.add_argument('--max_gen_length', default=256, type=int,
                        help='The maximum length of the sequence to be generated.')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='The value used to module the next token probabilities.')
    parser.add_argument('--k', default=0, type=int,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument('--p', default=0, type=float,
                        help='If set to float < 1, only the most probable tokens with probabilities that add up to '
                             'top_p or higher are kept for generation.')
    parser.add_argument('--num_beams', default=1, type=int, required=False, help='beams for beam search')
    parser.add_argument('--do_sample', action='store_true',
                        help='Whether to generate predictions via sampling; if off, decoding is done greedily.')

    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="max length of considered token sequences")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Do evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay applied to model parameters.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--patience', default=-1, type=int, required=False,
                        help='Number of epochs without dev-set loss improvement to ignore before '
                             'stopping the training.')

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps; unused, as models are saved after each epoch")
    parser.add_argument('--eval_epochs', type=int, default=100,
                        help="eval every X updates steps; unused, as models are evaluated after each epoch")
    parser.add_argument('--max_save_checkpoints', type=int, default=500,
                        help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and "
                             "ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed used for initialization of model parameters")
    parser.add_argument("--save_total_limit", type=int, default=0,
                        help="How many checkpoints to keep; if set to 0, keep all checkpoints")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Determine adapter insertion sites
    args = parser.parse_args()

    # Determine the adapter FFN size
    args.adapter_ffn_size = args.adapter_size * 4

    from_layers = 'from_layer' if args.initialize_adapters_from_layers else 'from_scratch'
    name_prefix = \
        'maxlen-' + str(args.max_seq_length) + '_' + 'batch-' + str(args.per_gpu_train_batch_size) + \
        '_' + 'lr-' + str(args.learning_rate) + '_' + 'warmup-' + str(args.warmup_steps) + '_' + 'epoch-' + \
        str(args.num_train_epochs)
    adapter_list = args.adapter_list if args.adapter_list is not None else 'none'
    plm_only = 'sequential_injection' if args.plm_only else 'adapter_injection'
    args.my_model_name = plm_only + '_' + '_'.join(args.active_domains) + '_' + args.fact_format + '_' + from_layers + '_' + \
        name_prefix + '_' + adapter_list
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Setup distant debugging, if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logging.info('Waiting for debugger attach ...')
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes / GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning('Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Instantiate model and tokenizer (NOTE: Currently designed to work with BART only)
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    pretrained_model = PretrainedModel(args)
    # Add new (special) tokens to the tokenizer and model embeddings
    new_tokens = ['<|D|>', '<|S|>', '<|R|>', '<|O|>']
    tokens_to_add = list()
    tokenizer_vocab = tokenizer.get_vocab()
    # Check if tokens exist in tokenizer vocab
    for tok in new_tokens:
        if tokenizer_vocab.get(tok, None) is None:
            tokens_to_add.append(tok)
    if len(tokens_to_add) > 0:
        # Add to tokenizer vocab
        tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
        # Initialize new embeddings
        # adapter_model uses the same output projection as the pretrained model
        pretrained_model.model.resize_token_embeddings(len(tokenizer))

    def _load_model(_model, model_file_path=None):
        """ Helper function for loading the adapter model checkpoints """
        model_dict = _model.state_dict()
        if model_file_path is None:
            model_file_path = args.adapter_model_checkpoint
        model_file_path = os.path.join(model_file_path, 'pytorch_model.bin')
        logger.info('Loading model state dict from {}'.format(model_file_path))
        found_adapter_model_parameters = torch.load(model_file_path, map_location=lambda storage, loc: storage)
        logger.info('Loading pretrained adapter model parameters')
        model_dict.update(found_adapter_model_parameters)
        _model.load_state_dict(model_dict)
        return _model

    if not args.plm_only:
        adapter_model = AdapterModel(args, pretrained_model)
        # Load adapter model parameters
        if args.adapter_model_checkpoint is not None:
            adapter_model = _load_model(adapter_model)
        adapter_model.to(args.device)
    else:
        if args.adapter_model_checkpoint is not None:
            pretrained_model = _load_model(pretrained_model)
        adapter_model = None

    pretrained_model.to(args.device)
    models = (pretrained_model, adapter_model)

    # Training
    best_checkpoint_path = None
    if args.do_train:
        train_datasets = load_and_cache_examples(args, tokenizer)  # train_datasets is a dictionary
        for domain in train_datasets.keys():
            global_step, tr_loss, best_checkpoint_path = \
                train(args, train_datasets[domain][0], models, tokenizer, domain)
            logger.info(" Finished training: global step = {}, mean loss per update = {}".format(global_step, tr_loss))

    # Save final model
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Take care of distributed/parallel training
        if adapter_model is not None:
            model_to_save = adapter_model.module if hasattr(adapter_model, 'module') else adapter_model
        else:
            model_to_save = pretrained_model.model.module if hasattr(pretrained_model.model, 'module') else \
                pretrained_model.model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    if args.do_eval:
        model = pretrained_model if args.plm_only else adapter_model
        # Load best model
        if args.do_train:
            model = _load_model(model, best_checkpoint_path)
        else:
            model = _load_model(model)
        for domain in args.active_domains:
            evaluate(args, model, tokenizer, domain, epoch=0, step=0, show_low_bleu_seqs=True)


if __name__ == '__main__':
    main()
