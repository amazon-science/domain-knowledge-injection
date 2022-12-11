import json
import torch
import random
import argparse
from word2number import w2n

from transformers import BartTokenizer, BartForConditionalGeneration

# General zero responses found in test/dialogues_001/2.json of MultiWOZ
NULL_RESPONSES = ["I'm sorry but I have not found any matches.",
                  "I'm sorry I don't have that information.",
                  "Sorry, I'm not finding anything.",
                  "I'm sorry, our system seems to be missing that information.",
                  "I'm sorry. I don't have that information.",
                  "I'm sorry, there isn't anything that matches those requirements.",
                  "I am sorry but there is nothing matching your request.",
                  "Sorry, I don't see any listings that match your search.",
                  "I'm sorry, our system does not have that information.",
                  "Sorry, there was nothing found."]

DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'bus', 'hospital', 'police', 'taxi']

# NOTE: Zero-responses ended up not being used in the final KPRS dataset, but the functionality to include them has been
# left in place for potential future work


def _collect_distractors(databases):
    """ Collects all permissible values for each domain-specific slot for distractor entity lookup. """
    slot_values = dict()

    # Iterate over domain entries and collect slot values
    print('Collecting distractors ...')
    for domain in databases.keys():
        slot_values[domain] = dict()
        for entry in databases[domain]:
            # Exclude locations and hotel prices, as they are not relevant to the dialogues
            for slot in entry.keys():
                if type(entry[slot]) != str:
                    continue
                slot_key = slot.replace(' ', '').lower()
                if slot_values[domain].get(slot_key, None) is None:
                    slot_values[domain][slot_key] = set()
                entry_slot_value = entry[slot].strip()
                if entry_slot_value != '?' and entry_slot_value not in slot_values[domain][slot_key]:
                    slot_values[domain][slot_key].add(entry_slot_value)

    # Convert sets to lists for JSON serialization
    for domain in slot_values.keys():
        for slot_key in slot_values[domain].keys():
            slot_values[domain][slot_key] = list(slot_values[domain][slot_key])

    return slot_values


def _get_model_ppl(context, response):
    """ Computes model perplexity of the response conditioned on the dialogue context. """

    # Create inputs and target
    context_string = ''
    for turn in context:
        if len(context_string) > 0:
            context_string += ' '
        context_string += '{}: {}'.format(turn[0]['speaker'], turn[0]['utterance'].strip())
    input_sequence = '{} SYSTEM: {}'.format(context_string, tokenizer.mask_token)
    target_sequence = '{} SYSTEM: {}'.format(context_string, response.strip())
    inputs = tokenizer([input_sequence], return_tensors='pt')
    for k, v in inputs.items():
        inputs[k] = v.to(torch.device(device))
    targets = tokenizer([target_sequence], return_tensors='pt')['input_ids'].to(torch.device(device))

    # Compute perplexity
    outputs = model(**inputs, labels=targets)
    loss = outputs.loss
    return torch.exp(loss).item()


def _create_negative_responses(positive_response, dialogue_history, distractors):
    """ Derives a negative response from the specified positive (i.e. true / correct) response
    and samples a null-response """

    perturbed_responses = list()
    sampled_null_responses = list()
    mod_types = list()

    # Compute 'filter-model' perplexity for the true response
    positive_ppl = _get_model_ppl(dialogue_history, positive_response['utterance'])

    # Determine possible perturbations, based on dialogue act
    # If multiple items of information are provided, create a negative sample for each of them
    act_table = positive_response['dialog_act']
    for full_act in act_table.keys():
        domain, act = full_act.split('-')
        domain, act = domain.strip().lower(), act.strip().lower()
        # Only consider relevant dialogue acts
        if act not in ['inform', 'recommend', 'nooffer'] or domain in ['booking', 'taxi']:
            continue

        # In cases where the user query can not be answered with DB information, the sample is skipped
        if positive_response['info_not_in_kb']:
            continue
        else:
            for slot, value in act_table[full_act]:

                picked_distractor = None
                blacklisted_distractors = set()  # distractors resulting in ineffective negative samples
                keep_negative_response = False
                value = str(value)

                if positive_response['utterance'].count(value) != 1:
                    continue

                while not keep_negative_response:
                    if slot == 'choice':
                        # Increment original value by some small amount
                        try:
                            num_choices = int(value.strip())
                        except ValueError:
                            try:
                                num_choices = w2n.word_to_num(value.strip().lower())
                                # For consistency, replace the number word in the original utterance with the
                                # corresponding integer
                                positive_response['utterance'] = \
                                    positive_response['utterance'].replace(value, str(num_choices))
                                value = str(num_choices)
                            except ValueError:
                                break  # skip non-numeric choices

                        # Exit loop if value is not present in the response or if it is present more than once
                        if positive_response['utterance'].count(str(value)) != 1:
                            break

                        choices = [n for n in range(int(num_choices) + 10) if
                                   (n != int(num_choices) and str(n) not in blacklisted_distractors)]
                        if len(choices) > 0:
                            random.shuffle(choices)
                            picked_distractor = str(choices[0])

                    elif slot == 'phone':
                        # Remove non-integer elements
                        phone = list()
                        for ch in list(value):
                            try:
                                phone.append(int(ch))
                            except ValueError:
                                phone.append(ch)
                        # Pick a digit at random and replace it with another random digit
                        new_phone = phone[:]
                        new_phone_string = ''.join([str(n) for n in new_phone])
                        blacklisted_distractors.add(new_phone_string)
                        stale_loops = 0
                        while new_phone_string in blacklisted_distractors:

                            stale_loops += 1
                            if stale_loops >= 100:
                                break  # exit inner loop

                            new_phone = phone[:]
                            digit_pos = random.choice(range(len(new_phone)))
                            while type(new_phone[digit_pos]) != int:
                                digit_pos = random.choice(range(len(new_phone)))
                            digit_range = [n for n in range(10) if n != new_phone[digit_pos]]
                            new_phone[digit_pos] = random.choice(digit_range)
                            new_phone_string = ''.join([str(n) for n in new_phone])

                        if stale_loops >= 100:
                            break  # Exit the outer loop

                        picked_distractor = new_phone_string
                        blacklisted_distractors.add(new_phone_string)

                    else:
                        # Exclude slots for which no KB information exists (e.g. 'ref')
                        if slot.lower() not in distractors[domain].keys():
                            break
                        if value not in distractors[domain][slot.lower()]:
                            break
                        # Do not perturb values that have been mentioned previously (as otherwise the model could prefer
                        # the correct reply because it is consistent with dialogue context)
                        skip = False
                        for turn in dialogue_history:
                            if value.lower() in turn[0]['utterance'].lower():
                                skip = True
                                break
                        if skip:
                            break

                        # Look-up initial distractor set
                        distractor_set = \
                            set([v for v in distractors[domain][slot.lower()] if v.lower() != value.lower()])
                        # Refine distractor set so that it doesn't include potentially correct distractors
                        # Look at most recent KB results and eliminate distractors supported by the retrieved entities
                        db_state = dialogue_history[-1][-1]  # This accesses the KB results
                        for entity in db_state:
                            intersection = list(set(entity) & distractor_set)
                            for dist in intersection:
                                distractor_set.remove(dist)
                        distractor_set = list(distractor_set)

                        # Sample a random distractor from the refined set
                        if len(distractor_set) > 0:
                            if slot.lower() == 'leaveat':
                                arrive_val = None
                                for tpl in act_table[full_act]:
                                    if tpl[0] == 'arriveby':
                                        try:
                                            arrive_val = int(tpl[1].replace(':', ''))
                                        except ValueError:
                                            continue
                                if arrive_val is not None:
                                    valid_distractors = \
                                        [d for d in distractor_set if int(d.replace(':', '')) < arrive_val]
                                    if len(valid_distractors) > 0:
                                        random.shuffle(valid_distractors)
                                        for dist in valid_distractors:
                                            if positive_response['utterance'].count(dist) == 0 and \
                                                    dist not in blacklisted_distractors:
                                                picked_distractor = dist
                                                break

                            if slot.lower() == 'arriveby':
                                leave_val = None
                                for tpl in act_table[full_act]:
                                    if tpl[0] == 'leaveat':
                                        try:
                                            leave_val = int(tpl[1].replace(':', ''))
                                        except ValueError:
                                            continue
                                if leave_val is not None:
                                    valid_distractors = \
                                        [d for d in distractor_set if int(d.replace(':', '')) > leave_val]
                                    if len(valid_distractors) > 0:
                                        random.shuffle(valid_distractors)
                                        for dist in valid_distractors:
                                            if positive_response['utterance'].count(dist) == 0 \
                                                    and dist not in blacklisted_distractors:
                                                picked_distractor = dist
                                                break

                            if picked_distractor is None:
                                random.shuffle(distractor_set)
                                for dist in distractor_set:
                                    if positive_response['utterance'].count(dist) == 0 \
                                            and dist not in blacklisted_distractors:
                                        picked_distractor = dist
                                        break

                    if picked_distractor is None:
                        break  # exit the outer while-loop

                    # Perturb positive sample
                    else:
                        # Modify determiners to maintain grammaticality
                        response_copy = positive_response['utterance']
                        if ' a {}'.format(value) in response_copy and \
                                picked_distractor[0].lower() in ['a', 'e', 'i', 'o', 'u']:
                            response_copy = response_copy.replace('a {}'.format(value), 'an {}'.format(value))
                        if 'A {}'.format(value) in response_copy and \
                                picked_distractor[0].lower() in ['a', 'e', 'i', 'o', 'u']:
                            response_copy = \
                                    response_copy.replace('A {}'.format(value), 'An {}'.format(value))
                        if ' an {}'.format(value) in response_copy and \
                                picked_distractor[0].lower() not in ['a', 'e', 'i', 'o', 'u']:
                            response_copy = response_copy.replace('an {}'.format(value), 'a {}'.format(value))
                        if 'An {}'.format(value) in response_copy and \
                                picked_distractor[0].lower() not in ['a', 'e', 'i', 'o', 'u']:
                            response_copy = \
                                    response_copy.replace('An {}'.format(value), 'A {}'.format(value))

                        negative_utterance = response_copy.replace(value, picked_distractor)
                        if positive_response['utterance'].lower() != negative_utterance.lower():
                            # Recompute the modified positive sample (inefficient)
                            if slot == 'choice':
                                positive_ppl = _get_model_ppl(dialogue_history, positive_response['utterance'])

                            # Compute 'filter-model' perplexity for the constructed response
                            # If it's higher than that of the true response, create another negative sample
                            negative_ppl = _get_model_ppl(dialogue_history, negative_utterance)
                            keep_negative_response = True if negative_ppl < positive_ppl else False

                            if not keep_negative_response:
                                blacklisted_distractors.add(picked_distractor)
                                picked_distractor = None
                            else:
                                perturbed_responses.append(negative_utterance)
                                # Sample a null-response for each perturbed response
                                null_response = random.choice(NULL_RESPONSES)
                                sampled_null_responses.append(null_response)
                                mod_types.append((domain, slot))

    return perturbed_responses, sampled_null_responses, mod_types


def create_samples(contexts_path, database_dir, distractors_path, create_only_positives):
    """ Filters contexts for the final dataset, identifies distractors, and generates contrastive samples """

    # Read-in contexts
    with open(contexts_path, 'r', encoding='utf8') as cp_in:
        contexts = json.load(cp_in)

    # Read-in json database files
    databases = dict()
    for domain in DOMAINS:
        if domain == 'taxi':
            continue
        db = '{}/{}_db.json'.format(database_dir, domain)
        with open(db, 'r', encoding='utf8') as in_f:
            databases[domain] = json.load(in_f)
        databases['taxi'] = dict()  # taxi database is not really a database

    # Read-in or collect distractors
    if distractors_path is None:
        distractors = _collect_distractors(databases)
        distractors_path = '/'.join(contexts_path.split('/')[:-1] + ['/distractors.json'])
        with open(distractors_path, 'w', encoding='utf8') as dist_f:
            json.dump(distractors, dist_f, indent=3, sort_keys=True, ensure_ascii=False)
    else:
        with open(distractors_path, 'r', encoding='utf8') as dist_f:
            distractors = json.load(dist_f)

    # Initialize containers
    single_domain_samples = dict()
    multi_domain_samples = dict()

    # Track stats
    num_unique_dialogues = set()
    num_unique_contexts = 0
    num_true_null_samples = 0
    num_non_null_samples = 0
    perturbation_counts_single_domain = {d: dict() for d in DOMAINS + ['no_domain']}
    perturbation_counts_multi_domain = {d: dict() for d in DOMAINS + ['no_domain']}
    samples_per_domain_single = {d: 0 for d in DOMAINS + ['no_domain']}
    samples_per_domain_multi = dict()

    # Iterate through contexts
    print('Generating samples ...')
    for dia_id in contexts.keys():
        dia_domains = contexts[dia_id]['services']
        for sample in contexts[dia_id]['contexts']:
            context = sample['context']
            true_response = sample['true_response']

            # Check if sample requires knowledge of multiple domains
            context_domains = set()
            for turn_id, turn in enumerate(context):
                turn_domains = [k.split('-')[0].lower() for k in turn[0]['dialog_act'].keys() if
                                ('inform' in k.lower() or 'recommend' in k.lower())]
                for dom in turn_domains:
                    if dom not in context_domains and dom != 'booking':
                        context_domains.add(dom)

            # Some samples have no assigned domain
            if len(context_domains) == 0:
                context_domains = {'no_domain'}

            if not create_only_positives:
                # Get negative responses
                perturbed_responses, null_responses, mod_types = \
                    _create_negative_responses(true_response, context, distractors)

                if len([pr for pr in perturbed_responses if pr is not None]) > 0:
                    num_unique_contexts += 1
                    if dia_id not in num_unique_dialogues:
                        num_unique_dialogues.add(dia_id)

                # Construct benchmark sample and store based on number of dialogue domains
                for pr_id, pr in enumerate(perturbed_responses):
                    if pr is None:
                        continue

                    bm_sample = {k: v for k, v in sample.items()}
                    bm_sample['false_response'] = pr
                    bm_sample['null_response'] = null_responses[pr_id]
                    bm_sample['perturbed_slot'] = mod_types[pr_id]

                    sample_set = single_domain_samples if len(context_domains) == 1 else multi_domain_samples
                    if sample_set.get(dia_id, None) is None:
                        sample_set[dia_id] = dict()
                        sample_set[dia_id]['dialog_domains'] = dia_domains
                        sample_set[dia_id]['samples'] = list()
                    bm_sample['sample_domains'] = list(context_domains)
                    sample_set[dia_id]['samples'].append(bm_sample)
                    if len(context_domains) == 1:
                        samples_per_domain_single[list(context_domains)[0]] += 1
                    else:
                        sorted_doms = ','.join(sorted(list(context_domains)))
                        if samples_per_domain_multi.get(sorted_doms, None) is None:
                            samples_per_domain_multi[sorted_doms] = 0
                        samples_per_domain_multi[sorted_doms] += 1

                    # Update trackers
                    if mod_types[pr_id] == 'true_null':
                        num_true_null_samples += 1
                    else:
                        num_non_null_samples += 1
                    perturbation_counts = perturbation_counts_single_domain if len(context_domains) == 1 else \
                        perturbation_counts_multi_domain
                    if perturbation_counts[mod_types[pr_id][0]].get(mod_types[pr_id][1], None) is None:
                        perturbation_counts[mod_types[pr_id][0]][mod_types[pr_id][1]] = 0
                    perturbation_counts[mod_types[pr_id][0]][mod_types[pr_id][1]] += 1

            else:
                bm_sample = {k: v for k, v in sample.items()}
                bm_sample['false_response'] = None
                bm_sample['null_response'] = None
                bm_sample['perturbed_slot'] = None

                sample_set = single_domain_samples if len(context_domains) == 1 else multi_domain_samples
                if sample_set.get(dia_id, None) is None:
                    sample_set[dia_id] = dict()
                    sample_set[dia_id]['dialog_domains'] = dia_domains
                    sample_set[dia_id]['samples'] = list()
                bm_sample['sample_domains'] = list(context_domains)
                sample_set[dia_id]['samples'].append(bm_sample)

                # Update trackers
                num_non_null_samples += 1
                num_unique_dialogues.add(dia_id)
                num_unique_contexts += 1

                if len(context_domains) == 1:
                    samples_per_domain_single[list(context_domains)[0]] += 1
                else:
                    sorted_doms = ','.join(sorted(list(context_domains)))
                    if samples_per_domain_multi.get(sorted_doms, None) is None:
                        samples_per_domain_multi[sorted_doms] = 0
                    samples_per_domain_multi[sorted_doms] += 1

    # Create .jsonl format samples
    single_domain_samples_jsonl = list()
    multi_domain_samples_jsonl = list()
    single_domain_utterances_jsonl = list()
    multi_domain_utterances_jsonl = list()
    for sample_set, jsonl, utterances_jsonl in \
            [(single_domain_samples, single_domain_samples_jsonl, single_domain_utterances_jsonl),
             (multi_domain_samples, multi_domain_samples_jsonl, multi_domain_utterances_jsonl)]:
        for dia_id in sample_set.keys():
            for sample_id, sample in enumerate(sample_set[dia_id]['samples']):
                sample_id = dia_id + '-{}'.format(sample_id)
                jsonl.append({'sample_id': sample_id,
                              'dialog_domains': sample_set[dia_id]['dialog_domains'],
                              'sample': sample})
                context_utterances = list()
                for turn in sample['context']:
                    context_utterances.append('{}: {}'.format(turn[0]['speaker'], turn[0]['utterance']))
                sample_utterances = {k: v for k, v in sample.items()}
                sample_utterances['context'] = context_utterances
                sample_utterances['true_response'] = sample_utterances['true_response']['utterance']
                utterances_jsonl.append(sample_utterances)

    # Write to file
    print('-' * 10)
    print('Saving samples ...')
    out_dir_path = '/'.join(contexts_path.split('/')[:-1])
    single_domain_path_json = out_dir_path + '/single_domain_samples.json'
    single_domain_path_jsonl = out_dir_path + '/single_domain_samples.jsonl'
    single_domain_utterances_path_jsonl = out_dir_path + '/single_domain_utterances.jsonl'

    multi_domain_path_json = out_dir_path + '/multi_domain_samples.json'
    multi_domain_path_jsonl = out_dir_path + '/multi_domain_samples.jsonl'
    multi_domain_utterances_path_jsonl = out_dir_path + '/multi_domain_utterances.jsonl'

    for collection, path in [(single_domain_samples, single_domain_path_json),
                             (single_domain_samples_jsonl, single_domain_path_jsonl),
                             (single_domain_utterances_jsonl, single_domain_utterances_path_jsonl),
                             (multi_domain_samples, multi_domain_path_json),
                             (multi_domain_samples_jsonl, multi_domain_path_jsonl),
                             (multi_domain_utterances_jsonl, multi_domain_utterances_path_jsonl)]:
        if not path.endswith('.jsonl'):
            with open(path, 'w', encoding='utf8') as out_f:
                json.dump(collection, out_f, indent=3, sort_keys=True, ensure_ascii=False)
        else:
            with open(path, 'w', encoding='utf8') as out_f:
                for line_id, line in enumerate(collection):
                    if line_id > 0:
                        out_f.write('\n')
                    out_f.write(json.dumps(line))
        print('Saved {} contrastive samples to {}'.format(len(single_domain_samples_jsonl) if 'single' in path else
                                                          len(multi_domain_samples_jsonl), path))

    # Report stats
    samples_total = num_non_null_samples + num_true_null_samples
    print('=' * 20)
    print('Kept {} unique dialogues'.format(len(num_unique_dialogues)))
    print('Kept {} unique contexts'.format(num_unique_contexts))
    print('Created {} samples overall; {} with non-null true responses, {} with true null responses'.format(
        samples_total, num_non_null_samples, num_true_null_samples))
    print('Created {:.2f} samples per context, on average; {:.2f} per dialogue'.format(
        samples_total / num_unique_contexts, samples_total / len(num_unique_dialogues)))
    print('-' * 10)
    print('Samples per domain (all | single | multi):')
    for domain in DOMAINS + ['no_domain']:
        single_counts = samples_per_domain_single[domain]
        multi_counts = 0
        # multi-domain samples require knowledge of multiple domains for processing the entire context
        # (i.e. not just final user statement)
        for key, value in samples_per_domain_multi.items():
            if domain in key:
                multi_counts += value
        print('\t{} : {} | {} | {}'.format(domain, single_counts + multi_counts, single_counts, multi_counts))
    print('-' * 10)
    print('Samples per perturbation type (all / single / multi):')
    for domain in perturbation_counts_single_domain.keys():
        for slot in list(set(list(perturbation_counts_single_domain[domain].keys()) +
                             list(perturbation_counts_multi_domain[domain].keys()))):
            single_counts = perturbation_counts_single_domain[domain].get(slot, 0)
            multi_counts = perturbation_counts_multi_domain[domain].get(slot, 0)
            print('\t{}-{} : {} | {} | {}'.format(
                domain, slot, single_counts + multi_counts, single_counts, multi_counts))
        print('\t' + ('-' * 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contexts_path', type=str, required=True,
                        help='path pointing to the dialogue contexts file')
    parser.add_argument('--database_dir', type=str, required=True,
                        help='path to the database directory')
    parser.add_argument('--distractors_path', type=str, default=None,
                        help='path to the file containing distractor entities for the creation of negative samples')
    parser.add_argument("--only_positives", action='store_true',
                        help="Whether to generate negative responses")
    args = parser.parse_args()

    # Initialize transformer model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    model.to(torch.device(device))

    create_samples(args.contexts_path, args.database_dir, args.distractors_path, args.only_positives)
