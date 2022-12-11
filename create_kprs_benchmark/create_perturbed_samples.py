# General idea:
# 1. Perturb databases using perturb_databases.py
# 2. Collect perturbed contexts using collect_contexts.py and the perturbed databases
# 3. Iterate through contexts
# 4. For each context, look up appropriate entities in the perturbed DBs
# 5. Compare set of perturbed entities with the original set of entities and make sure they have no overlap
# 6. Treat the original response as the false alternative
# 7. Construct the correct response by replacing the communicated value in the original response with the value
# supplied by one of the perturbed entities (value can be either entity name or requestable)


import json
import argparse

from collect_contexts import _query_json_db

DOMAINS = ['restaurant', 'hotel', 'attraction', 'train']


# TODO / NOTE: By construction, the permuted responses may be disfluent (although we try to control for that),
#  which will cause model errors if models favour fluency over factuality


def _create_permuted_response(original_response,
                              turn,
                              original_databases,
                              perturbed_databases,
                              perturbation_mapping,
                              perturbed_domain):

    """ Derives a response that is supported by the permuted DBs from the original response. """

    domain = None
    perturbed_responses = list()
    original_span_info = original_response['span_info']
    for tpl in original_span_info:
        domain, act = tpl[0].split('-')
        domain, act = domain.strip().lower(), act.strip().lower()
        # Only perturb responses corresponding to the domain with the perturbed DB (i.e. 'restaurant' by default)
        if domain != perturbed_domain:
            continue

        # Query perturbed DBs for entities that are relevant to the current system response
        original_hits, _, _, _ = _query_json_db(domain, turn[0], original_response['utterance'], original_databases)
        perturbed_hits, _, _, _ = _query_json_db(domain, turn[0], original_response['utterance'], perturbed_databases)
        # Reject responses that can't be permuted (i.e. are not relevant to the DBs)
        if len(original_hits) == 0:
            continue

        # For recommendation responses, replace the entity name with the entity corresponding to the same DB entry
        # in the perturbed DB
        if act == 'recommend':
            if tpl[1] == 'name':
                # Check if the original entity is consistent with the perturbed database
                skip_sample = any([entity['name'].lower() == tpl[1].lower() for entity in perturbed_hits])

                if not skip_sample:
                    if tpl[2].lower() not in perturbation_mapping.keys():
                        continue
                    # Construct the perturbed response by replacing the original entity
                    replacement_name = \
                        ' '.join([w.capitalize() for w in perturbation_mapping[tpl[2].lower()].split()])
                    perturbed_response = original_response['utterance'].replace(tpl[2], replacement_name)
                    perturbed_responses.append((perturbed_response, 'request'))
                    break

        # For inform responses, replace requestables (name, address, and phone) based on the entry corresponding to
        # the relevant entity in the perturbed DB
        if act == 'inform':
            if tpl[1] in ['name', 'address', 'phone'] and len(perturbed_responses) == 0:
                perturbed_response = original_response['utterance']
                # Construct the perturbed reply by replacing the original requestable
                # 1. Find entry that the original info corresponds to in the original DB
                # (names, addresses, and phones are unique)
                # 2. Look up mapping for the permuted DB
                # 3. Replace all applicable information in the original response with the permuted info,
                # to make the whole reply consistent with the permuted DB
                relevant_original_entry = None

                for entry in original_databases[domain]:
                    if tpl[1].lower() not in entry.keys():
                        continue
                    if entry[tpl[1].lower()].lower() == tpl[2].lower() or \
                            entry[tpl[1].lower()].lower() == tpl[2].lower().replace(' ', ''):
                        relevant_original_entry = entry
                        break

                if relevant_original_entry is not None:
                    possible_matches = \
                        [entity for entity in perturbed_databases[domain] if entity['name'] ==
                         relevant_original_entry['name'].lower()] + \
                        [entity for entity in perturbed_databases[domain] if entity['name'] ==
                         relevant_original_entry['name']]
                    if len(possible_matches) == 0:
                        continue
                    perturbed_entry = possible_matches[0]

                    # Replace all requestables occurring in the original response to be consistent with the perturbed DB
                    for span in original_span_info:
                        if span[1] not in relevant_original_entry.keys():
                            continue
                        if type(relevant_original_entry[span[1]]) != str:
                            continue
                        if span[1] not in perturbed_entry.keys():
                            continue
                        if perturbed_entry[span[1]] in perturbed_response:
                            continue  # avoid double replacements
                        perturbed_response = perturbed_response.replace(span[2], perturbed_entry[span[1]])

                    if perturbed_response != original_response['utterance']:
                        perturbed_responses.append((perturbed_response, 'inform'))
                        break  # only create one sample with all relevant requestables replaced

    if len(perturbed_responses) == 0:
        perturbed_responses.append((original_response['utterance'], 'no_change'))

    # Prioritize name_only substitutions, as they are more likely to be grammatical
    if len(perturbed_responses) > 1:
        for rsp in perturbed_responses:
            if rsp[1] == 'request':
                return rsp[0], rsp[1]

    return perturbed_responses[0][0], perturbed_responses[0][1]


def create_samples(contexts_path,
                   original_db_dir,
                   perturbed_db_dir,
                   perturbation_mapping,
                   create_only_positives,
                   perturbed_domain):

    """ Creates a perturbed KPRS benchmark variant based on perturbed databases """

    # Read-in contexts
    print('Read-in contexts ...')
    with open(contexts_path, 'r', encoding='utf8') as cp_in:
        contexts = json.load(cp_in)

    # Read-in json database files (originals and perturbed versions)
    print('Read-in databases ...')
    original_databases, perturbed_databases = dict(), dict()
    for domain in DOMAINS:
        if domain == 'taxi':
            continue
        db = '{}/{}_db.json'.format(original_db_dir, domain)
        with open(db, 'r', encoding='utf8') as in_f:
            original_databases[domain] = json.load(in_f)
        db = '{}/{}_db.json'.format(perturbed_db_dir, domain)
        with open(db, 'r', encoding='utf8') as in_f:
            perturbed_databases[domain] = json.load(in_f)
        original_databases['taxi'] = dict()  # taxi database is not really a database
        perturbed_databases['taxi'] = dict()
    # Read in the mapping between entities in the original and perturbed domains
    with open(perturbation_mapping, 'r', encoding='utf8') as in_f:
        mapping = json.load(in_f)
        mapping = {key.lower(): value for key, value in mapping.items()}

    # Initialize containers
    single_domain_samples = dict()
    multi_domain_samples = dict()

    # Track stats
    num_samples = 0
    num_unique_dialogues = set()
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

            perturbed_response, mod_type = _create_permuted_response(true_response,
                                                                     context[-1],
                                                                     original_databases,
                                                                     perturbed_databases,
                                                                     mapping,
                                                                     perturbed_domain)
            original_response = true_response['utterance']

            # Create benchmark samples and update perturbation counts
            bm_sample = {k: v for k, v in sample.items()}
            bm_sample['false_response'] = original_response if not create_only_positives else None
            bm_sample['true_response']['utterance'] = perturbed_response
            sample_set = single_domain_samples if len(context_domains) == 1 else multi_domain_samples
            perturbation_counts = perturbation_counts_single_domain if len(context_domains) == 1 else \
                perturbation_counts_multi_domain
            domain_str = '|'.join(list(context_domains))
            if domain_str not in perturbation_counts.keys():
                perturbation_counts[domain_str] = dict()
            if mod_type not in perturbation_counts[domain_str].keys():
                perturbation_counts[domain_str][mod_type] = 0
            perturbation_counts[domain_str][mod_type] += 1

            if sample_set.get(dia_id, None) is None:
                sample_set[dia_id] = dict()
                sample_set[dia_id]['dialog_domains'] = dia_domains
                sample_set[dia_id]['samples'] = list()
            bm_sample['sample_domains'] = list(context_domains)
            sample_set[dia_id]['samples'].append(bm_sample)
            if len(context_domains) == 1:
                if list(context_domains)[0] not in samples_per_domain_single.keys():
                    continue
                samples_per_domain_single[list(context_domains)[0]] += 1
            else:
                sorted_doms = ','.join(sorted(list(context_domains)))
                if samples_per_domain_multi.get(sorted_doms, None) is None:
                    samples_per_domain_multi[sorted_doms] = 0
                samples_per_domain_multi[sorted_doms] += 1

            # Update trackers
            num_samples += 1
            num_unique_dialogues.add(dia_id)

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

    # Write the perturbed benchmark to file
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
    print('=' * 20)
    print('Kept {} unique dialogues'.format(len(num_unique_dialogues)))
    print('Created {} samples overall'.format(num_samples))
    print('Created {:.2f} samples per dialogue, on average'.format(num_samples / len(num_unique_dialogues)))
    print('-' * 10)
    print('Samples per domain (all | single | multi):')
    for domain in DOMAINS + ['no_domain']:
        single_counts = samples_per_domain_single[domain]
        multi_counts = 0
        for key, value in samples_per_domain_multi.items():
            if domain in key:
                multi_counts += value
        print('\t{} : {} | {} | {}'.format(domain, single_counts + multi_counts, single_counts, multi_counts))
    print('-' * 10)
    print('Samples per perturbation type, single-domain (related to the perturbed domain):')
    for item in perturbation_counts_single_domain[perturbed_domain].items():
        print(perturbed_domain, item)
    print('\n\n')
    print('Samples per perturbation type, multi-domain (related to the perturbed domain):')
    for domain in perturbation_counts_multi_domain.keys():
        if perturbed_domain in domain:
            for item in perturbation_counts_multi_domain[domain].items():
                print(domain, item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contexts_path', type=str, required=True,
                        help='path pointing to the dialogue contexts file')
    parser.add_argument('--original_db_dir', type=str, required=True,
                        help='path to the original database directory')
    parser.add_argument('--perturbed_db_dir', type=str, required=True,
                        help='path to the perturbed database directory')
    parser.add_argument('--perturbation_mapping', type=str, required=True,
                        help='path to the entity mapping denoting the database perturbations')
    parser.add_argument("--only_positives", action='store_true',
                        help="Whether to generate negative responses")
    parser.add_argument('--perturbed_domain', type=str, default='restaurant',
                        help='domain of the perturbed database')
    args = parser.parse_args()

    create_samples(args.contexts_path,
                   args.original_db_dir,
                   args.perturbed_db_dir,
                   args.perturbation_mapping,
                   args.only_positives,
                   args.perturbed_domain)
