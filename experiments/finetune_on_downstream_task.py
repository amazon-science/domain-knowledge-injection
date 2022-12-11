import os
import sys
import json
import torch
import random
import logging
import argparse
import sacrebleu

import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartTokenizer

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from adapter_model import PretrainedModel, AdapterModel
from util import PROCESSORS, DS_PROCESSORS, convert_examples_to_features_ds, _rotate_checkpoints, \
    load_mwoz_databases_json

from evaluate_kprs import evaluate_model_on_krgs
from evaluate_response_generation import evaluate_all_samples as evaluate_generation


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
logger = logging.getLogger(__name__)


# TODO: Is this compatible with the sequential-finetuning baseline?

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
    adapter_model = models[1]  # == None if args.plm_only
    main_model = pretrained_model if adapter_model is None else adapter_model

    # Set batch size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Handle data serving
    train_sampler = RandomSampler(train_dataset[0]) if args.local_rank == -1 else DistributedSampler(train_dataset[0])
    train_dataloader = DataLoader(train_dataset[0], sampler=train_sampler, batch_size=args.train_batch_size)

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
        if adapter_model is not None:
            adapter_model, optimizer = amp.initialize(adapter_model, optimizer, opt_level=args.fp16_opt_level)
        else:
            pretrained_model, optimizer = amp.initialize(pretrained_model, optimizer, opt_level=args.fp16_opt_level)

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
        adapter_model = torch.nn.parallel.DistributedDataParallel(adapter_model,
                                                                  device_ids=[args.local_rank],
                                                                  output_device=args.local_rank)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = {}".format(len(train_dataset[0])))
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
    dev_loss, dev_acc = None, None
    best_dev_loss, best_dev_acc = float('inf'), 0.
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
            if adapter_model is not None:
                adapter_model.eval()
            pretrained_model.train()  # Freeze parameters of the adapters(s)

            # Perform a single forward pass
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}
            # Prepare decoder inputs and labels for enc-dec models
            inputs['labels'] = batch[3][:, 1:].clone().contiguous()  # shift
            decoder_input_ids = batch[3][:, :-1].clone().contiguous()  # shift
            decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id  # remove masking
            inputs['decoder_input_ids'] = decoder_input_ids.contiguous()
            inputs['decoder_attention_mask'] = batch[4][:, :-1].clone().contiguous()

            if adapter_model is not None:
                outputs = adapter_model(None, current_domain, **inputs, return_dict=False)
            else:
                outputs = pretrained_model(**inputs, return_dict=True)

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
        if epoch_id % args.eval_epochs == 0 or args.eval_epochs == 1:

            if args.local_rank in [-1, 0]:
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.local_rank == -1 and args.evaluate_during_training:
                    dev_loss, dev_acc = evaluate(args, [pretrained_model, adapter_model], tokenizer,
                                                 current_domain, epoch_id, global_step)

            # Save model checkpoint
            if args.local_rank in [-1, 0]:
                checkpoint_path = \
                    os.path.join(args.output_dir, 'checkpoint-{}_adapter-task-{}-epoch_{}-gs_{}'.format(
                    '_'.join(args.active_domains), args.task, epoch_id, global_step))
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
                    _rotate_checkpoints(args, 'checkpoint-{}_adapter-task-{}'.format(
                        '_'.join(args.active_domains), args.task), use_mtime=False)

            # Check whether to stop training early
            save_best = False
            if dev_acc is not None and best_dev_acc < dev_acc:
                best_dev_acc = dev_acc
                save_best = True
            if dev_loss is not None and best_dev_loss > dev_loss:
                best_dev_loss = dev_loss
                save_best = True
            if save_best:
                stale_epochs = 0
                best_checkpoint_path = \
                    os.path.join(args.output_dir, 'checkpoint-{}_adapter-task-{}-best'.format(
                        '_'.join(args.active_domains), args.task))
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
                    '!!! Performance has not improved this epoch. Stale epochs: {} !!!'.format(stale_epochs))

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


def evaluate(args, models, tokenizer, domain, epoch=0, step=0):
    """ Checks model performance on dev set """

    # Reference previous evaluation results
    results = dict()
    if os.path.exists(os.path.join(args.output_dir, 'dev_metrics.json')):
        with open(os.path.join(args.output_dir, 'dev_metrics.json'), 'r') as f:
            existing_results = json.loads(f.read())
        f.close()
        results.update(existing_results)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    loss, accuracy, bleu = None, None, None
    if args.task == 'response_selection':
        accuracy = evaluate_model_on_krgs(args.kprs_dev_path, models, tokenizer, domain,
                                          plm_only=args.plm_only, device=args.device)

    else:
        # Designate models
        pretrained_model = models[0]
        adapter_model = models[1]  # == None if args.plm_only

        # Set batch size
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Set up data-serving pipeline
        dev_dataset = load_and_cache_examples(args, tokenizer, mode='dev')
        dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        dev_sampler = SequentialSampler(dev_dataset[domain][0]) if args.local_rank == -1 \
            else DistributedSampler(dev_dataset[domain][0])
        dev_dataloader = DataLoader(dev_dataset[domain][0], sampler=dev_sampler, batch_size=dev_batch_size)

        # Validate!
        logger.info("***** Running validation *****")
        logger.info("  Num dev examples = {}".format(len(dev_dataset)))
        logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_eval_batch_size))

        # Iterate over update steps
        epoch_losses = []
        dev_iterator = trange(int(1), desc='Epoch', disable=args.local_rank not in [-1, 0])
        for epoch_id, _ in enumerate(dev_iterator):
            epoch_iterator = \
                tqdm(dev_dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0], mininterval=10, ncols=100)
            for step, batch in enumerate(epoch_iterator):
                if adapter_model is not None:
                    adapter_model.eval()
                pretrained_model.eval()

                # Perform a single forward pass
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1]}
                # Prepare decoder inputs and labels for enc-dec models
                inputs['labels'] = batch[3][:, 1:].clone().contiguous()  # shift
                decoder_input_ids = batch[3][:, :-1].clone().contiguous()  # shift
                decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id  # remove masking
                inputs['decoder_input_ids'] = decoder_input_ids.contiguous()
                inputs['decoder_attention_mask'] = batch[4][:, :-1].clone().contiguous()

                if adapter_model is not None:
                    outputs = adapter_model(None, domain, **inputs, return_dict=False)
                else:
                    outputs = pretrained_model(**inputs, return_dict=True)

                # Loss is obtained from the full model
                # model outputs are always tuple in pytorch-transformers (see doc)
                step_loss = outputs[0]
                epoch_losses.append(step_loss.item())

        loss = np.mean(epoch_losses)

    # Update results
    if results.get('epoch_and_step', None) is None:
        results['epoch_and_step'] = []
    if results.get('mean_dev_loss', None) is None:
        results['mean_dev_loss'] = []
    if results.get('mean_dev_accuracy', None) is None:
        results['mean_dev_accuracy'] = []
    if results.get('mean_dev_bleu', None) is None:
        results['mean_dev_bleu'] = []

    results['epoch_and_step'].append((epoch, step))
    results['mean_dev_loss'].append(loss)
    results['mean_dev_accuracy'].append(accuracy)
    results['mean_dev_bleu'].append(bleu)

    output_eval_file = \
        os.path.join(args.output_dir, 'dev_results-{}_adapter-task-{}-epoch_{}-gs_{}'.format(
            domain, args.task, epoch, step))
    with open(output_eval_file, 'w') as writer:
        logger.info('***** Evaluation results *****')
        writer.write('STEP: {:s}\n'.format(str(step)))
        writer.write('%s = %s\n' % ('mean dev loss', str(loss)))
        writer.write('%s = %s\n' % ('mean dev accuracy', str(accuracy)))
        writer.write('%s = %s\n' % ('mean dev BLEU', str(bleu)))

    logger.info('  %s = %s', 'mean dev loss', str(loss))
    logger.info('  %s = %s', 'mean dev accuracy', str(accuracy))
    logger.info('  %s = %s', 'mean dev BLEU', str(bleu))

    # Maintain a single metrics file
    with open(os.path.join(args.output_dir, 'dev_metrics.json'), 'w') as f:
        f.write(json.dumps(results))

    return loss, accuracy


def test(args, models, tokenizer, domain, epoch=0, step=0, mode='test'):
    """ Checks the success of knowledge injection by letting the model generate masked DB information """

    test_dataset = load_and_cache_examples(args, tokenizer, mode=mode)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset[domain][0]) if args.local_rank == -1 \
        else DistributedSampler(test_dataset[domain][0])
    test_dataloader = DataLoader(test_dataset[domain][0], sampler=test_sampler, batch_size=test_batch_size)
    all_requires_db, all_domains, all_contents = \
        test_dataset[domain][1], test_dataset[domain][2], test_dataset[domain][3]
    sample_guids = test_dataset[domain][-1]

    # Test!
    logger.info('***** Testing on held-out data *****')
    logger.info('  Num examples = %d', len(test_dataset[domain]))
    logger.info('  Batch size = %d', test_batch_size)
    generation_inputs, generation_targets, generated_sequences = list(), list(), list()

    # Designate models
    pretrained_model = models[0]
    adapter_model = models[1]  # == None if args.plm_only
    model = adapter_model if adapter_model is not None else pretrained_model.model

    # Iterate through the test corpus
    model.eval()
    for batch_id, batch in enumerate(tqdm(test_dataloader, desc='Testing', mininterval=10, ncols=100)):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():

            input_ids = batch[0]
            attention_mask = batch[1]
            target_ids = batch[3]
            gen_prompt = tokenizer.eos_token_id

            max_gen_length = args.max_gen_length
            outputs = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     min_length=5,
                                     max_length=max_gen_length,
                                     temperature=args.temperature,
                                     top_k=args.k if args.k > 0 else None,
                                     top_p=args.p if args.p > 0 else None,
                                     num_beams=args.num_beams,  # greedy decoding
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

    # assert len(generation_inputs) == len(generation_targets) == len(generated_sequences) == len(all_requires_db) \
    #        == len(all_domains),\
    #     'Generation output size mismatch! {}, {}, {}, {}'.format(len(generation_inputs), len(generation_targets),
    #                                                              len(generated_sequences), len(all_requires_db))

    # Report sample generation results
    logging.info('***** Example generations *****')
    for s_id, gen_input in enumerate(generation_inputs):
        if s_id >= 10:
            break
        logging.info('-' * 10)
        logging.info('  Input: {:s}'.format(gen_input))
        logging.info('  Target: {:s}'.format(generation_targets[s_id]))
        logging.info('  Prediction: {:s}'.format(generated_sequences[s_id]))
        logging.info('  Required DB access: {}'.format(all_requires_db[s_id]))

    # Compute and update evaluation metric values
    task_acc = 0.0
    bleu = 0.0
    partial_accuracies = None

    if args.task == 'state_tracking':

        # Compute corpus-level BLEU
        bleu = sacrebleu.corpus_bleu(generated_sequences, [[t] for t in generation_targets]).score

        # Read in databases and check all possible slot value combinations
        databases = load_mwoz_databases_json(args.database_dir)
        new_databases = {dom: dict() for dom in databases.keys()}
        for domain in databases.keys():
            for entry in databases[domain]:
                if 'name' in entry.keys():
                    new_databases[domain][entry['name']] = {k.lower().replace(' ', ''): v for k, v in entry.items()}
                if 'trainID' in entry.keys():
                    new_databases[domain][entry['trainID']] = {k.lower().replace(' ', ''): v for k, v in entry.items()}
        databases = new_databases

        all_slot_values = {domain: {} for domain in ['restaurant', 'hotel', 'attraction', 'train', 'mixed']}
        for domain in databases.keys():
            for entry in databases[domain].keys():
                for slot in databases[domain][entry].keys():
                    if type(databases[domain][entry][slot]) != str:
                        continue
                    if slot not in databases[domain][entry].keys():
                        continue
                    if slot not in all_slot_values[domain].keys():
                        all_slot_values[domain][slot] = list()
                    if slot not in all_slot_values['mixed'].keys():
                        all_slot_values['mixed'][slot] = list()
                    if databases[domain][entry][slot].lower() not in all_slot_values[domain][slot]:
                        all_slot_values[domain][slot].append(databases[domain][entry][slot].lower())
                    if databases[domain][entry][slot].lower() not in all_slot_values['mixed'][slot]:
                        all_slot_values['mixed'][slot].append(databases[domain][entry][slot].lower())


        # Split targets into individual slot segments and check if they are in the generated response
        turns = {'num_correct': 0, 'num_incorrect': 0}
        slot_value_combos_tokens = {'valid': 0, 'invalid': 0}
        slot_value_combos_types = {'valid': list(), 'invalid': list()}
        for t_id, t in enumerate(generation_targets):
            ref_slots = [seg.strip() for seg in t.split(', ')
                         if
                         seg.strip().split(' ')[0].strip() in ['restaurant', 'hotel', 'attraction', 'train', 'mixed']]
            gen_slots = [seg.strip() for seg in generated_sequences[t_id].split(', ')
                         if
                         seg.strip().split(' ')[0].strip() in ['restaurant', 'hotel', 'attraction', 'train', 'mixed']]
            slot_hits = [slot in generated_sequences[t_id] for slot in ref_slots]
            if all(slot_hits) and len(ref_slots) == len(gen_slots):
                turns['num_correct'] += 1
            else:
                turns['num_incorrect'] += 1
            for slot in gen_slots:
                slot_domain = slot.split()[0].lower()
                slot_value_pair = slot.split()[1:]
                if len(slot_value_pair) < 2:
                    continue
                slot_id = slot_value_pair[0]
                slot_value = ' '.join(slot_value_pair[1:])
                if slot_domain not in all_slot_values.keys():
                    continue
                else:
                    if slot_id not in all_slot_values[slot_domain]:
                        if 'book' in slot_id:
                            continue
                        else:
                            slot_value_combos_tokens['invalid'] += 1
                            if slot_value not in slot_value_combos_types['invalid']:
                                slot_value_combos_types['invalid'].append(slot_value)
                            continue
                    if slot_value in all_slot_values[slot_domain][slot_id]:
                        slot_value_combos_tokens['valid'] += 1
                        if slot_value not in slot_value_combos_types['valid']:
                            slot_value_combos_types['valid'].append(slot_value)
                    else:
                        slot_value_combos_tokens['invalid'] += 1
                        if slot_value not in slot_value_combos_types['invalid']:
                            slot_value_combos_types['invalid'].append(slot_value)

        joint_goal_accuracy = turns['num_correct'] / (turns['num_correct'] + turns['num_incorrect'])
        validity_ratio_tokens = \
            slot_value_combos_tokens['valid'] / (
                        slot_value_combos_tokens['valid'] + slot_value_combos_tokens['invalid'])
        validity_ratio_types = len(slot_value_combos_types['valid']) / \
                               (len(slot_value_combos_types['valid']) + len(slot_value_combos_types['invalid']))
        print('Joint Goal Accuracy: {}'.format(joint_goal_accuracy))
        print('# valid slot combos tokens: {}, # invalid slot combos: {}, validity ratio: {}'.format(
            slot_value_combos_tokens['valid'], slot_value_combos_tokens['invalid'], validity_ratio_tokens))
        print('# valid slot combos types: {}, # invalid slot combos: {}, validity ratio: {}'.format(
            len(slot_value_combos_types['valid']), len(slot_value_combos_types['invalid']), validity_ratio_types))
        task_acc = joint_goal_accuracy

    # Evaluate generated responses
    if args.task == 'response_generation':

        # Load test samples from JSON file
        samples_path = \
            'test/merged_samples.json' if len(args.active_domains) > 1 else 'test/single_domain_samples.json'
        with open(os.path.join(args.data_dir, samples_path), 'r', encoding='utf8') as f:
            samples = json.load(f)

        # Evaluate model generations
        generated_sequences_dict = {sample_guids[idx]: sequence for idx, sequence in enumerate(generated_sequences)}
        inform_acc, request_acc, choice_acc, no_offer_acc, inform_noref_count, mean_inform_ratios, \
            mean_request_ratios, inform_metrics, request_metrics = evaluate_generation(samples,
                                                                                       generated_sequences_dict,
                                                                                       args.database_dir,
                                                                                       requires_db_only=True)

        # Task accuracy is computed as the average of inform accuracy and request accuracy
        # But all metrics are reported nonetheless
        task_acc = (inform_acc + request_acc) / 2
        logging.info('Response generation metrics: ')
        logging.info('   Inform accuracy: {:.4f}'.format(inform_acc))
        logging.info('   Inform NoRef count: {:d}'.format(inform_noref_count))
        logging.info('   Request accuracy: {:.4f}'.format(request_acc))
        logging.info('   Choice accuracy: {:.4f}'.format(choice_acc))
        logging.info('   No offer accuracy: {:.4f}'.format(no_offer_acc))
        logging.info('   Inform metrics: {}'.format(inform_metrics))
        logging.info('   Request metrics: {}'.format(request_metrics))

    # Compute corpus BLEU
    generation_targets_nested = [generation_targets]
    corpus_bleu = sacrebleu.corpus_bleu(generated_sequences, generation_targets_nested).score

    # Log metrics
    output_eval_file = os.path.join(args.output_dir,
                                    'generation_test_results-{}_adapter-epoch_{}-gs_{}'.format(domain, epoch, step))
    with open(output_eval_file, 'w') as writer:
        writer.write('STEP: {:s}\n'.format(str(step)))
        writer.write('%s = %s\n' % ('task accuracy', str(task_acc)))
        writer.write('%s = %s\n' % ('corpus-BLUE', str(bleu) + ' | ' + str(corpus_bleu)))
        if partial_accuracies is not None:
            writer.write('%s = %s\n' % ('partial accuracies', str(partial_accuracies)))

    logger.info('***** Evaluation results *****')
    logger.info('  %s = %s', 'task accuracy', str(task_acc))
    logger.info('  %s = %s', 'corpus-BLEU', str(bleu) + ' | ' + str(corpus_bleu))
    if partial_accuracies is not None:
        logger.info('  %s = %s', 'partial accuracies', str(partial_accuracies))

    # Log predictions
    output_pred_file = os.path.join(args.output_dir,
                                    'generation_test_predictions-{}_adapter-epoch_{}-gs_{}'.format(domain, epoch, step))
    with open(output_pred_file, 'w') as writer:
        logger.info('***** Write predictions *****')
        for gsi, gs in enumerate(generated_sequences):
            writer.write(json.dumps({'input': generation_inputs[gsi],
                                     'target': generation_targets[gsi],
                                     'prediction': gs}) + '\n')

    return bleu, task_acc


# ======================================================================================================================
# Data preprocessing

def load_and_cache_examples(args, tokenizer, mode='train'):
    """ Pre-process raw inputs for training the model """

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Select the right processor
    if args.task != 'composite_facts_eval':
        processor = DS_PROCESSORS[args.task](tokenizer, args)
    else:
        processor = PROCESSORS[args.task](tokenizer, args)
    # Load data features from cache or dataset file
    cached_features_file = \
        os.path.join(args.data_dir, 'cached_{}_{}_{}_{}.bin'.format('_'.join(args.active_domains),
                                                                    args.fact_format, str(args.max_seq_length),
                                                                    mode))
    if os.path.exists(cached_features_file):
        # Load existing features
        logger.info('Loading features from cached file {}'.format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        # Generate sample features from raw input
        logger.info('Creating features from dataset file at {}'.format(args.data_dir))
        # Load samples
        if mode == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif mode == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        # Generate features
        features = convert_examples_to_features_ds(examples, args.max_seq_length, tokenizer,
                                                   mix_samples=len(args.active_domains) > 1 or args.active_domains[0] == 'mixed',
                                                   hard_max_len=args.task == 'response_generation')  # returns a dictionary
        # Save features for future retrieval
        if args.local_rank in [-1, 0]:
            logger.info('Saving features into cached file {}'.format(cached_features_file))
            torch.save(features, cached_features_file)

    # Convert features to tensors and build dataset
    all_datasets = dict()
    for domain in features.keys():
        if domain in args.active_domains or domain == 'mixed':
            all_input_ids = torch.tensor([f.input_ids for f in features[domain]], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features[domain]], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features[domain]], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in features[domain]], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in features[domain]], dtype=torch.long)
            all_domains = [f.domain for f in features[domain]]
            all_need_db = [f.need_db for f in features[domain]]
            all_contents = [f.contents for f in features[domain]]

            # These will be none for most tasks
            all_slots = [f.slot for f in features[domain]]
            all_slot_seq = [f.slot_seq for f in features[domain]]
            all_entities = [f.entity for f in features[domain]]

            all_guids = [f.guid for f in features[domain]]  # used primarily for response generation experiments

            all_datasets[domain] = \
                (TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_target_ids, all_target_mask),
                 all_need_db, all_domains, all_contents, all_slots, all_slot_seq, all_entities, all_guids)
    return all_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--database_dir", default=None, type=str,
                        help="The database directory, required for the evaluation of generated responses")
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
    parser.add_argument("--adapter_model_checkpoint", type=str, nargs='+', default=None,
                        help="Path to the checkpoints of the pre-trained adapter-enhanced models")
    parser.add_argument("--model_checkpoint", type=str, nargs='+', default=None,
                        help="Path to the checkpoints of the pre-trained sequentially fine-tuned model")
    parser.add_argument("--adapter_combo_method", type=str, default='concatenate',
                        help="Format of the DB facts presented to the model")
    parser.add_argument("--task", type=str, choices=['response_selection', 'response_generation', 'state_tracking',
                                                     'composite_facts_eval'],
                        required=True, help="Downstream task to fine-tune the model on")
    parser.add_argument("--plm_only", action='store_true',
                        help="Whether to fine-tune the pre=trained LM only, i.e. without taking adapters into account")
    parser.add_argument("--kprs_dev_path", type=str, default="path to the development KPRS split")
    parser.add_argument("--kprs_test_path", type=str, default="path to the test KPRS split")
    parser.add_argument("--clone_lm_head", action='store_true',
                        help="Whether to use a separate LM head for the downstream task.")
    parser.add_argument("--no_encoder_integration", action='store_true',
                        help="Disables adapter integration into the encoder output.")

    ## Generation parameters
    parser.add_argument('--max_gen_length', default=60, type=int,
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
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run testing on the test set.")
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

    name_prefix = \
        'maxlen-' + str(args.max_seq_length) + '_' + 'batch-' + str(args.per_gpu_train_batch_size) + \
        '_' + 'lr-' + str(args.learning_rate) + '_' + 'warmup-' + str(args.warmup_steps) + '_' + 'epoch-' + \
        str(args.num_train_epochs)
    adapter_list = args.adapter_list if args.adapter_list is not None else 'none'
    plm_only = 'plm_only' if args.plm_only else 'plm_plus_adapter'
    if args.adapter_model_checkpoint is None:
        plm_only = 'plm_plus_random_adapter'
    args.my_model_name = args.task + '_' + plm_only + '_' + '_'.join(args.active_domains) + '_' + args.fact_format + '_' + \
        args.adapter_combo_method + '_' + name_prefix + '_' + adapter_list
    if not args.no_encoder_integration:
        args.my_model_name += '_encoder_integration'
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

    def _add_tokens(token_list):
        """ Helper function for adding ne tokens to the model vocabulary """

        tokens_to_add = list()
        tokenizer_vocab = tokenizer.get_vocab()
        # Check if tokens exist in tokenizer vocab
        for tok in token_list:
            if tokenizer_vocab.get(tok, None) is None:
                tokens_to_add.append(tok)
        if len(tokens_to_add) > 0:
            # Add to tokenizer vocab
            tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
            # Initialize new embeddings
            # adapter_model uses the same output projection as the pretrained model
            pretrained_model.model.resize_token_embeddings(len(tokenizer))

    pretrained_model = PretrainedModel(args)
    # Add new (special) tokens to the tokenizer and model embeddings
    _add_tokens(['<|D|>', '<|S|>', '<|R|>', '<|O|>'])
    adapter_model = AdapterModel(args, pretrained_model) if not args.plm_only else None

    def _load_model(model_file_path=None, is_adapter_model=False):
        """ Helper function for loading the adapter model checkpoints """
        target_model = adapter_model if is_adapter_model else pretrained_model
        model_dict = target_model.state_dict()
        updated_keys = list()  # for debugging

        # Load different adapters
        logger.info('Loading pretrained adapter model parameters')
        if model_file_path is None:
            for ch_id, checkpoint in enumerate(args.adapter_model_checkpoint):
                model_file_path = os.path.join(checkpoint, 'pytorch_model.bin')
                logger.info('Loading model state dict from {}'.format(model_file_path))
                found_adapter_model_parameters = torch.load(model_file_path, map_location=lambda storage, loc: storage)
                if ch_id > 0:
                    # Identify which of the adapter model parameters do not match the pretrained adapter checkpoint
                    mismatched_param_keys = \
                        [key for key in model_dict.keys() if key not in found_adapter_model_parameters.keys()]
                    # Modify pretrained adapter model parameter names so that they can be loaded in
                    for load_key in mismatched_param_keys:
                        # Fetch pretrained values
                        if 'encoders.{}.'.format(ch_id) in load_key:
                            fetch_key = load_key.replace('encoders.{}.'.format(ch_id), 'encoders.{}.'.format(0))
                            found_adapter_model_parameters[load_key] = found_adapter_model_parameters[fetch_key]
                            del found_adapter_model_parameters[fetch_key]
                            updated_keys.append(load_key)
                        elif 'decoders.{}.'.format(ch_id) in load_key:
                            fetch_key = load_key.replace('decoders.{}.'.format(ch_id), 'decoders.{}.'.format(0))
                            found_adapter_model_parameters[load_key] = found_adapter_model_parameters[fetch_key]
                            del found_adapter_model_parameters[fetch_key]
                            updated_keys.append(load_key)
                        else:
                            continue

                model_dict.update(found_adapter_model_parameters)
        target_model.load_state_dict(model_dict)

    if not args.plm_only:
        # Load adapter model parameters
        if args.adapter_model_checkpoint is not None:
            _load_model(is_adapter_model=True)
        adapter_model.to(args.device)
    else:
        # Load pre-trained model for sequential fine-tuning
        if args.model_checkpoint is not None:
            _load_model()

    # Add new tokens for the down-stream tasks
    _add_tokens(['<|DOM|>', '<|INT|>', '<|REQ|>', '<|SLO|>', '<|SEP|>'])
    pretrained_model.to(args.device)
    models = (pretrained_model, adapter_model)

    # Training
    best_checkpoint_path = None
    if args.do_train:
        train_datasets = load_and_cache_examples(args, tokenizer, mode='train')  # train_datasets is a dictionary
        domain = args.active_domains[0] if len(args.active_domains) == 1 else 'mixed'
        global_step, tr_loss, best_checkpoint_path = train(args, train_datasets[domain], models, tokenizer, domain)
        logger.info(" Finished training: global step = {}, mean loss per update = {}".format(global_step, tr_loss))

    # Save final model
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Take care of distributed/parallel training
        if args.plm_only:
            model_to_save = \
                pretrained_model.model.module if hasattr(pretrained_model, 'module') else pretrained_model.model
        else:
            model_to_save = adapter_model.module if hasattr(adapter_model, 'module') else adapter_model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    if args.do_eval:
        domain = args.active_domains[0] if len(args.active_domains) == 1 else 'mixed'
        if args.task in ['state_tracking',  'response_generation']:
            test_result = test(args, [pretrained_model, adapter_model], tokenizer, domain, 0, 0)
        else:
            test_result = evaluate_model_on_krgs(args.kprs_test_path,
                                                 [pretrained_model, adapter_model],
                                                 tokenizer,
                                                 args.active_domains,
                                                 plm_only=args.plm_only,
                                                 device=args.device)
        logging.info('*' * 10)
        logging.info('Evaluation concluded!')
        logging.info('Evaluation result: {}'.format(test_result))

    if args.do_test:
        # Load best model
        if args.do_train:
            if args.plm_only:
                _load_model(best_checkpoint_path)
            else:
                _load_model(best_checkpoint_path, is_adapter_model=True)
        else:
            if args.plm_only:
                _load_model()
            else:
                _load_model(is_adapter_model=True)

        domain = args.active_domains[0] if len(args.active_domains) == 1 else 'mixed'
        test(args, adapter_model, tokenizer, domain, epoch=0, step=0)


if __name__ == '__main__':
    main()
