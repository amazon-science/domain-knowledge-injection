import json
import argparse

from util import load_mwoz_databases

# Note: This is a slightly modified variant of the KPRS construction script
# Response generation data is created to include DB information, which may not be used by the knowledge-injected model

DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'bus', 'hospital', 'police', 'taxi']


def _query_json_db(domain, turn, sys_rep, dbs):
    """ Workaround for the 'hospital' DB which cannot be accessed with sqlite3; should return values of relevant entries
    as a tuple. Used for hospital and police domains. """

    # Restructuring the turn info
    slot_values, requested_slots = dict(), dict()
    for entry in turn['frames']:
        slot_values[entry['service']] = entry['state']['slot_values']
        requested_slots[entry['service']] = entry['state']['requested_slots']
        if entry['service'] == domain:
            if entry['state']['active_intent'] is None:
                return list(), 0, False, False

    # Remove booking slot values
    non_book_slot_values_keys = [key for key in slot_values[domain].keys() if 'book' not in key]
    non_book_requested_slots_keys = [s for s in requested_slots if 'book' not in s]
    if len(non_book_slot_values_keys) == 0 and len(non_book_requested_slots_keys) == 0:
        return list(), 0, False, False

    # Isolate relevant state entries
    hits = list()
    domain_slot_values = slot_values[domain]

    if domain == 'hospital':
        for key in domain_slot_values.keys():
            for entry in dbs[domain]:
                for val in domain_slot_values[key]:
                    if key in ['hospital-telephone', 'hospital-address', 'hospital-postcode'] and \
                            'department' in entry.keys():
                        continue
                    if key not in ['hospital-telephone', 'hospital-address', 'hospital-postcode'] and \
                            'department' not in entry.keys():
                        continue
                    if entry[key.split('-')[-1]] == val:
                        new_hit = sorted([entry[entry_key] for entry_key in entry.keys() if entry_key != 'id'])
                        if tuple(new_hit) not in hits:
                            hits.append(tuple(new_hit))
    else:
        # Police KB only has one entry
        police_keys = ['name', 'address', 'phone']
        hits.append((dbs[domain][0][pk] for pk in police_keys))

    num_entities = len(hits)
    # simple (but spotty) heuristic to identify correct KB retrieval failures
    true_null = num_entities == 0 and ('sorry' in sys_rep.lower() or 'unfortunately' in sys_rep.lower())
    failed_to_retrieve = num_entities == 0 and not true_null

    return hits, num_entities, failed_to_retrieve, true_null


def _query_result(domain, turn, sys_rep, dbs):
    """ Returns the list of entities for a given domain based on the annotation of the belief state
        (adopted from the MultiWOZ codebase) """
    # Query the DB
    sql_query = 'select * from {}'.format(domain)

    # Restructuring the turn info
    metadata = dict()
    for entry in turn['frames']:
        metadata[entry['service']] = entry['state']['slot_values']
        if entry['service'] == domain:
            if entry['state']['active_intent'] is None:
                return list(), 0, False, False

    # Remove booking slot values
    non_book_keys = [key for key in metadata[domain].keys() if 'book' not in key]
    if len(non_book_keys) == 0:
        return list(), 0, False, False

    flag = True
    for key in non_book_keys:
        val = metadata[domain][key]
        key = key.split('-')[-1]

        # Ignore underspecified requests
        if val[0] in ['dont care', 'dontcare', 'don\'t care']:
            continue

        use_or = False
        l_par = '(' if len(val) > 1 else ''
        r_par = ')' if len(val) > 1 else ''

        for v_id, v in enumerate(val):
            if v is None:
                pass
            else:
                connector = 'or' if use_or else 'and'
                if flag:
                    sql_query += " where "

                    if len(val) > 1 and v_id == 0:
                        sql_query += l_par

                    v2 = v.replace("'", "''")
                    # change query for trains
                    if key == 'leaveat':
                        sql_query += r" " + key + " > " + r"'" + v2 + r"'"
                    elif key == 'arriveby':
                        sql_query += r" " + key + " < " + r"'" + v2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + v2 + r"'"
                    flag = False
                else:
                    v2 = v.replace("'", "''")

                    if len(val) > 1 and v_id == 0:
                        if key == 'leaveat':
                            sql_query += r" {} ".format(connector) + ' {}'.format(l_par) + key + " > " + r"'" + v2 + r"'"
                        elif key == 'arriveby':
                            sql_query += r" {} ".format(connector) + ' {}'.format(l_par) + key + " < " + r"'" + v2 + r"'"
                        else:
                            sql_query += r" {} ".format(connector) + ' {}'.format(l_par) + key + "=" + r"'" + v2 + r"'"
                    else:
                        if key == 'leaveat':
                            sql_query += r" {} ".format(connector) + key + " > " + r"'" + v2 + r"'"
                        elif key == 'arriveby':
                            sql_query += r" {} ".format(connector) + key + " < " + r"'" + v2 + r"'"
                        else:
                            sql_query += r" {} ".format(connector) + key + "=" + r"'" + v2 + r"'"
                use_or = True

        if len(val) > 1:
            sql_query += ' {}'.format(r_par)

    dbs[domain].execute(sql_query)
    hits = dbs[domain].fetchall()
    # Remove ID, as it is not relevant to the dialogue
    if domain in ['restaurant', 'hotel', 'attraction']:
        hits = [hit[1:] for hit in hits]

    num_entities = len(hits)
    # simple (but spotty) heuristic to identify correct KB retrieval failures
    true_null = num_entities == 0 and ('sorry' in sys_rep.lower() or 'unfortunately' in sys_rep.lower())
    failed_to_retrieve = num_entities == 0 and not true_null

    return hits, num_entities, failed_to_retrieve, true_null


def collect_contexts(dialogue_files, dialogue_acts, database_dir, out_path):
    """ Collects all viable dialogue contexts (and subsequent system turns) to be used in the creation of the
    response generation data. """

    # Load-in databases
    databases = load_mwoz_databases(database_dir)
    print('Read-in {} databases'.format(len(databases.keys())))

    # Read-in dialogues and assign to each turn the corresponding dialogue act annotation
    all_dialogues = list()
    for dialogue_file in dialogue_files:
        with open(dialogue_file, 'r', encoding='utf8') as dia_in:
            all_dialogues.append(json.load(dia_in))
    with open(dialogue_acts, 'r', encoding='utf8') as act_in:
        acts = json.load(act_in)
    dialogues_plus_acts = dict()
    total_system_turns = 0
    for dialogues in all_dialogues:
        for dia in dialogues:
            dia_id = dia['dialogue_id']
            curr_act = acts[dia_id]
            assert len(dia['turns']) == len(curr_act), \
                'Dialogue acts annotation length ({}) does not match the dialogue length ({})'\
                    .format(len(dia['turns']), len(curr_act))
            dialogues_plus_acts[dia_id] = {'services': dia['services'], 'turns': dict()}
            for turn_id, turn in enumerate(dia['turns']):
                if turn['speaker'] == 'SYSTEM':
                    total_system_turns += 1
                dialogues_plus_acts[dia_id]['turns'][str(turn_id)] = turn
                for act_key in curr_act[str(turn_id)].keys():
                    dialogues_plus_acts[dia_id]['turns'][str(turn_id)][act_key] = curr_act[str(turn_id)][act_key]
    # Report
    print('Collected {} dialogues, containing {} system turns'.format(len(dialogues_plus_acts.keys()),
                                                                      total_system_turns))

    # Collect dialogue contexts for system turns that contain relevant dialogue acts
    # Each context is the full dialogue history up to the system turn (but the system turn itself is also stored)
    dialogue_contexts = dict()
    num_unique_contexts = 0
    for dia_id in dialogues_plus_acts.keys():
        dia_domains = dialogues_plus_acts[dia_id]['services']
        updated_context = list()
        contexts = list()
        has_failed_kb_call, is_true_null = False, False
        for turn in dialogues_plus_acts[dia_id]['turns'].keys():
            if dialogues_plus_acts[dia_id]['turns'][turn]['speaker'] == 'SYSTEM':

                # Check if any of the dialogue acts matches the requirements
                dia_acts = dialogues_plus_acts[dia_id]['turns'][turn]['dialog_act']
                requires_db = False
                for key in dia_acts.keys():
                    key = key.strip().lower()
                    if ('inform' in key or 'recommend' in key or 'nooffer' in key) and 'booking' not in key:
                        requires_db = True
                        break
                # Add context
                true_response = {k: v for k, v in dialogues_plus_acts[dia_id]['turns'][turn].items()}
                true_response['info_not_in_kb'] = is_true_null

                # Identify information that has to be included in the response
                db_info = {}
                for span_tpl in true_response['span_info']:
                    if span_tpl[1] not in db_info.keys():
                        db_info[span_tpl[1]] = list()
                    db_info[span_tpl[1]].append(span_tpl[2])
                true_response['required_db_info'] = db_info

                contexts.append({'context': updated_context[:],
                                 'response': true_response,
                                 'requires_db': requires_db})

            # Retrieve a DB state corresponding to the turn's belief state for each domain associated with the dialogue
            db_states = dict()
            is_true_null = False  # reset if it has been flipped to True on the previous turn
            if dialogues_plus_acts[dia_id]['turns'][turn]['speaker'] == 'USER':
                for domain in dia_domains:
                    if domain in ['hospital', 'police']:
                        db_state, _, failed_to_retrieve, is_true_null = \
                            _query_json_db(domain, dialogues_plus_acts[dia_id]['turns'][turn],
                                           dialogues_plus_acts[dia_id]['turns'][str(int(turn) + 1)]['utterance'],
                                           databases)
                        if len(db_state) > 0:
                            db_states[domain] = db_state
                        else:
                            db_states[domain] = None

                    elif domain in databases.keys():
                        db_state, _, failed_to_retrieve, is_true_null = \
                            _query_result(domain, dialogues_plus_acts[dia_id]['turns'][turn],
                                          dialogues_plus_acts[dia_id]['turns'][str(int(turn) + 1)]['utterance'],
                                          databases)
                        if len(db_state) > 0:
                            db_states[domain] = db_state
                        else:
                            db_states[domain] = None

                    else:
                        # Taxi domain, not sure how else to handle it
                        db_states[domain] = None

            updated_context.append((dialogues_plus_acts[dia_id]['turns'][turn], db_states))

        if len(contexts) > 0:
            dialogue_contexts[dia_id] = {'services': dialogues_plus_acts[dia_id]['services'], 'contexts': contexts}
            num_unique_contexts += len(contexts)

    # Report
    print('Kept {} relevant dialogues, isolated {} unique contexts'.format(len(dialogue_contexts.keys()),
                                                                           num_unique_contexts))

    # Write the collected contexts to disc
    with open(out_path, 'w', encoding='utf8') as out_f:
        json.dump(dialogue_contexts, out_f, indent=3, sort_keys=True, ensure_ascii=False)
    print('Saved collected contexts to {}'.format(out_path))

    return out_path


def create_samples(contexts_path):
    """ Creates response generation samples form the collected contexts. """

    # Read-in contexts
    with open(contexts_path, 'r', encoding='utf8') as cp_in:
        contexts = json.load(cp_in)

    # Initialize containers
    single_domain_samples = dict()
    multi_domain_samples = dict()

    # Track stats
    num_unique_dialogues = set()
    num_unique_contexts = 0
    samples_per_domain_single = {d: 0 for d in DOMAINS + ['no_domain']}
    samples_per_domain_multi = dict()

    # Iterate through contexts
    print('Generating samples ...')
    for dia_id in contexts.keys():
        dia_domains = contexts[dia_id]['services']
        for sample in contexts[dia_id]['contexts']:
            context = sample['context']

            # Check if sample requires knowledge of multiple domains
            context_domains = set()
            for turn in context:
                turn_domains = [k.split('-')[0].lower() for k in turn[0]['dialog_act'].keys() if
                                ('inform' in k.lower() or 'recommend' in k.lower())]
                for dom in turn_domains:
                    context_domains.add(dom)

            # Some samples have no assigned domain
            if len(context_domains) == 0:
                context_domains = {'no_domain'}

            gen_sample = {k: v for k, v in sample.items()}
            sample_set = single_domain_samples if len(context_domains) == 1 else multi_domain_samples
            if sample_set.get(dia_id, None) is None:
                sample_set[dia_id] = dict()
                sample_set[dia_id]['dialog_domains'] = dia_domains
                sample_set[dia_id]['samples'] = list()
            gen_sample['sample_domains'] = list(context_domains)
            sample_set[dia_id]['samples'].append(gen_sample)

            # Update trackers
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
                sample_utterances['response'] = sample_utterances['response']['utterance']
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
        print('Saved {} samples to {}'.format(len(single_domain_samples_jsonl) if 'single' in path else
                                              len(multi_domain_samples_jsonl), path))

    # Report stats
    print('=' * 20)
    print('Kept {} unique dialogues'.format(len(num_unique_dialogues)))
    print('Kept {} unique contexts / samples'.format(num_unique_contexts))
    print('-' * 10)
    print('Samples per domain (all | single | multi):')
    for domain in DOMAINS + ['no_domain']:
        single_counts = samples_per_domain_single[domain]
        multi_counts = 0
        # multi-domain samples require knowledge of multiple domains for processing the entire context
        # (i.e. not just final user statement)
        for key in samples_per_domain_multi.keys():
            if domain in key:
                multi_counts += samples_per_domain_multi[key]
        print('\t{} : {} | {} | {}'.format(domain, single_counts + multi_counts, single_counts, multi_counts))
    print('-' * 10)
    print('Multi-domain samples:')
    for key, value in samples_per_domain_multi.items():
        print('{} : {}'.format(key, value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogue_files', type=str, nargs='+', required=True,
                        help='list of paths to the files containing dialogues')
    parser.add_argument('--dialogue_acts', type=str, required=True,
                        help='path to the file containing the dialogue act annotations')
    parser.add_argument('--database_dir', type=str, required=True,
                        help='path to the database directory')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the file to which the collected contexts should be written')
    args = parser.parse_args()

    collect_contexts(args.dialogue_files, args.dialogue_acts, args.database_dir, args.out_path)
    create_samples(args.out_path)
