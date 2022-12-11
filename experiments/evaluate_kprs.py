import json
import torch
import argparse

import numpy as np

from transformers import BartForConditionalGeneration, BartTokenizer


def _compute_perplexity(token_probabilities):
    """ Computes perplexity on the basis of token-level probabilities """
    # Compute PPL from probabilities
    return np.prod([1 / p for p in token_probabilities]) ** (1. / len(token_probabilities))


def _score_response(model_input,
                    target,
                    models,
                    tokenizer,
                    allowed_domains,
                    device,
                    plm_only,
                    print_sample=False):
    """ Computes perplexity of contrastive translations """

    if print_sample:
        print('-' * 10)
        print('Input: {}'.format(model_input))
        print('Target: {}'.format(target))

    # Score target sequence conditioned on the input
    input_ids = tokenizer(model_input, return_tensors='pt')
    target_ids = tokenizer(target, return_tensors='pt').input_ids
    # Shift target IDs to create decoder inputs and targets
    decoder_input_ids = target_ids.clone().detach()[:, :-1].contiguous()  # shift
    decoder_target_ids = target_ids.clone().detach()[:, 1:].contiguous()  # shift

    if plm_only:
        model, adapter_model = models  # adapter_model == None in this case
        model.eval()
        # Do a forward pass
        with torch.no_grad():
            model_out = model(input_ids=input_ids.input_ids.to(device),
                              attention_mask=input_ids.attention_mask.to(device),
                              decoder_input_ids=decoder_input_ids.to(device),
                              labels=decoder_target_ids.to(device))
    else:
        pretrained_model, adapter_model = models
        pretrained_model.eval()
        adapter_model.eval()
        # Do a forward pass
        with torch.no_grad():
            model_out = adapter_model(None,
                                      allowed_domains,
                                      input_ids=input_ids.input_ids.to(device),
                                      attention_mask=input_ids.attention_mask.to(device),
                                      decoder_input_ids=decoder_input_ids.to(device),
                                      labels=decoder_target_ids.to(device),
                                      return_dict=False)

    # Extract probabilities
    decoder_target_list = torch.squeeze(decoder_target_ids).detach().cpu().numpy().tolist()
    model_logits = torch.squeeze(model_out[1])
    tgt_probs = model_logits.softmax(axis=-1).detach().cpu().numpy().tolist()
    token_probabilities = [tgt_probs[probs_row][tok_id] for probs_row, tok_id in enumerate(decoder_target_list)]

    return _compute_perplexity(token_probabilities)


def evaluate_model_on_krgs(json_file_paths,
                           models,
                           tokenizer,
                           allowed_domains,
                           plm_only=False,
                           device='cuda'):
    """ Checks model performance on the KPRS benchmark. The model is only provided dialogue utterances, without any
    meta-information such as dialogue states (for now). """

    # Standardize input
    if type(json_file_paths) != list:
        json_file_paths = [json_file_paths]
    if type(allowed_domains) != list:
        allowed_domains = [allowed_domains]

    # Read-in data
    all_samples = []
    for path in json_file_paths:
        with open(path, 'r', encoding='utf8') as jfp:
            all_samples.append(json.load(jfp))

    # Compute accuracy and track input features that may be indicative of the model using heuristics
    num_correct, num_incorrect = 0, 0
    correct_decisions, incorrect_decisions = list(), list()

    # Track the difference in perplexity between responses for correctly and incorrectly ranked pairs
    ppl_diff_correct_decision = list()
    ppl_diff_incorrect_decision = list()
    samples_seen = 0

    for samples in all_samples:
        for dialogue_id in samples.keys():
            for sample in samples[dialogue_id]['samples']:
                samples_seen += 1
                if allowed_domains != ['mixed']:
                    # Skip samples that pertain to excluded domains
                    skip_sample = False
                    for dom in sample['sample_domains']:
                        if dom not in allowed_domains:
                            skip_sample = True
                            break
                    if skip_sample:
                        continue

                # Report
                # print('Checking sample {:d}'.format(samples_seen))
                if samples_seen > 0 and samples_seen % 500 == 0:
                    print('Processed {:d} samples'.format(samples_seen))

                # Construct model input -- input and output are identical, with the input containing masked DB information
                # For now, input includes SYSTEM / USER designators, but those can be dropped in the future
                # NOTE: This evaluation paradigm can't make use of the 'null response' category

                # Collect context utterances
                context_string = ''
                for tpl in sample['context']:
                    context_string += '{}: {} '.format(tpl[0]['speaker'], tpl[0]['utterance'])

                # Construct response options
                true_response_string = 'SYSTEM: {}'.format(sample['true_response']['utterance'])
                false_response_string = 'SYSTEM: {}'.format(sample['false_response'])
                model_input = context_string
                true_target = true_response_string
                false_target = false_response_string

                # Score responses
                print_sample = samples_seen <= 10
                try:
                    true_target_ppl = _score_response(model_input, true_target, models, tokenizer, allowed_domains,
                                                      device, plm_only, print_sample)
                    false_target_ppl = _score_response(model_input, false_target, models, tokenizer, allowed_domains,
                                                       device, plm_only, print_sample)
                except ZeroDivisionError:
                    continue
                # tracks PPL difference between responses (higher is better)
                ppl_diff = false_target_ppl - true_target_ppl

                # print('>>> ', true_target_ppl, false_target_ppl, ppl_diff)

                if true_target_ppl < false_target_ppl:
                    num_correct += 1
                    correct_decisions.append((context_string, true_response_string, false_response_string))
                    ppl_diff_correct_decision.append(ppl_diff)
                else:
                    num_incorrect += 1
                    incorrect_decisions.append((context_string, true_response_string, false_response_string))
                    ppl_diff_incorrect_decision.append(ppl_diff)

    model_accuracy = num_correct / (num_correct + num_incorrect)
    # Report
    print('*' * 20)
    print('Evaluated {} samples for {} domain(s)'.format(num_correct + num_incorrect, allowed_domains))
    print('# of correct responses elicited lower model perplexity: {}'.format(num_correct))
    print('# of INcorrect responses elicited lower (or equal) model perplexity: {}'.format(num_incorrect))
    print('Model accuracy: {}'.format(num_correct / (num_correct + num_incorrect)))
    print('-' * 5)
    print('Mean perplexity difference between responses for correctly ranked samples: {}'.format(
        np.mean(ppl_diff_correct_decision)))
    print('Mean perplexity difference between responses for INcorrectly ranked samples: {}'.format(
        np.mean(ppl_diff_incorrect_decision)))
    print('*' * 20)

    return model_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_paths', nargs='+', type=str, required=True,
                        help='path to the JSON file containing the contrastive samples')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='path to the directory containing checkpoint files of the evaluated model')
    parser.add_argument('--allowed_domains', type=str, nargs='+', required=True,
                        help='list of dialogue domains that are evaluated by the script')
    parser.add_argument('--model_type', type=str, choices=['bart'],
                        help='Model type to evaluate')
    parser.add_argument('--use_cpu', action='store_true', help='Whether to use the CPU')
    args = parser.parse_args()

    # Assign checkpoints (change model size by changing checkpoint names)
    if args.model_type == 'bart':
        model_type = BartForConditionalGeneration
        tokenizer_type = BartTokenizer
    else:
        model_type = None
        tokenizer_type = None

    loaded_model = model_type.from_pretrained('facebook/bart-large')
    loaded_tokenizer = tokenizer_type.from_pretrained('facebook/bart-large')

    use_device = 'cpu' if args.use_cpu else 'cuda:0'
    loaded_model.to(use_device)

    evaluate_model_on_krgs(args.json_file_path, loaded_model, loaded_tokenizer, args.allowed_domains,
                           plm_only=True, device=use_device)
