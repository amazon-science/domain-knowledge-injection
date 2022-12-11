import json
import argparse

from util import load_mwoz_databases

# Booking and 'general' acts are excluded, as they are not relevant to the DB information
# Excluded domains: police, taxi, hospital (dialogues that include either of these domains will be skipped)
# Choice responses are kept only if they provide a numerical value (e.g. not 'several')

# All MultiWOZ dialogue acts:
# {'Hotel': ['Inform', 'Request', 'Recommend', 'Select', 'NoOffer'],
# 'Booking': ['Inform', 'NoBook', 'Request', 'Book'],
# 'general': ['reqmore', 'bye', 'thank', 'welcome', 'greet'],
# 'Police': ['Inform', 'Request'],
# 'Attraction': ['Inform', 'Request', 'Recommend', 'Select', 'NoOffer'],
# 'Train': ['Inform', 'Request', 'OfferBook', 'OfferBooked', 'NoOffer', 'Select'],
# 'Taxi': ['Inform', 'Request'],
# 'Restaurant': ['Inform', 'Request', 'NoOffer', 'Select', 'Recommend'],
# 'Hospital': ['Inform', 'Request']}

IDENTIFIERS = {'restaurant': 'name',
               'hotel': 'name',
               'attraction': 'name',
               'bus': 'trainID',
               'train': 'trainID',
               'hospital': 'department'}


def _query_json_db(domain, turn, sys_resp, dbs):
    """ Queries .JSON databases for relevant entries. """

    # Restructuring the turn info
    slot_values, requested_slots = dict(), dict()
    for entry in turn['frames']:
        if 'state' not in entry.keys():
            continue
        slot_values[entry['service']] = entry['state']['slot_values']
        requested_slots[entry['service']] = entry['state']['requested_slots']
        if entry['service'] == domain:
            if entry['state']['active_intent'] is None:
                return list(), 0, False, False

    if len(slot_values.keys()) == 0:
        return list(), 0, False, False
    if domain not in slot_values.keys():
        return list(), 0, False, False

    # Remove booking slot values
    non_book_slot_values_keys = [key for key in slot_values[domain].keys() if 'book' not in key]
    non_book_requested_slots_keys = [s for s in requested_slots if 'book' not in s]
    if len(non_book_slot_values_keys) == 0 and len(non_book_requested_slots_keys) == 0:
        return list(), 0, False, False

    # Isolate relevant state entries
    hits, hit_keys = list(), list()
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

    elif domain == 'police':
        # Police KB only has one entry
        police_keys = ['name', 'address', 'phone']
        hits.append((dbs[domain][0][pk] for pk in police_keys))

    else:
        key_to_entry = dict()
        or_hits = dict()
        for key in domain_slot_values.keys():
            slot = key.split('-')[-1]
            for entry in dbs[domain]:
                for val in domain_slot_values[key]:
                    if slot in entry.keys():
                        # Collect entries that match individual slot values
                        if entry[slot] == val:
                            if or_hits.get(slot, None) is None:
                                or_hits[slot] = list()
                            new_hit_key = ' | '.join(sorted([entry[entry_key] for entry_key in entry.keys() if
                                                             (entry_key != 'id' and type(entry[entry_key]) == str)]))
                            or_hits[slot].append(new_hit_key)
                            key_to_entry[new_hit_key] = entry
        # Check which entities satisfy all slot values
        hits = list()
        for key in key_to_entry.keys():
            match_all = all([key in or_hits[slot] for slot in or_hits.keys()])
            if match_all:
                hits.append(key_to_entry[key])

    num_entities = len(hits)
    # simple (but spotty) heuristic to identify correct KB retrieval failures
    true_null = num_entities == 0 and ('sorry' in sys_resp.lower() or 'unfortunately' in sys_resp.lower())
    failed_to_retrieve = num_entities == 0 and not true_null
    return hits, num_entities, failed_to_retrieve, true_null


def _query_result(domain, turn, sys_rep, dbs):
    """ Queries SQL databases for relevant entries; returns the list of entities for a given domain based on the
    annotation of the belief state (adopted from the MultiWOZ codebase) """

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
    # Simple (but spotty) heuristic to identify correct KB retrieval failures
    true_null = num_entities == 0 and ('sorry' in sys_rep.lower() or 'unfortunately' in sys_rep.lower())
    failed_to_retrieve = num_entities == 0 and not true_null
    return hits, num_entities, failed_to_retrieve, true_null


def collect_contexts(dialogue_files, dialogue_acts, database_dir, out_path):
    """ Collects all viable dialogue contexts (and subsequent system turns) to be used in the creation of the
    knowledge-guided response selection benchmark. """

    # Load-in databases
    databases = load_mwoz_databases(database_dir, all_json=True)  # all_json=True utilizes .json databases only
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
    print('Collected {} dialogues, containing {} system turns'.format(
        len(dialogues_plus_acts.keys()), total_system_turns))

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
                keep_sample = False
                for key in dia_acts.keys():
                    key = key.strip().lower()
                    if ('inform' in key or 'recommend' in key or 'nooffer' in key) and 'booking' not in key \
                            and 'taxi' not in key:
                        keep_sample = True
                        break
                if keep_sample:
                    # Add context
                    true_response = {k: v for k, v in dialogues_plus_acts[dia_id]['turns'][turn].items()}
                    true_response['info_not_in_kb'] = is_true_null
                    contexts.append({'context': updated_context[:], 'true_response': true_response})

            # Retrieve a DB state corresponding to the turn's belief state for each domain associated with the dialogue
            db_states = dict()
            is_true_null = False  # reset if it has been flipped to True on the previous turn
            if dialogues_plus_acts[dia_id]['turns'][turn]['speaker'] == 'USER':
                for domain in dia_domains:
                    db_state, _, failed_to_retrieve, is_true_null = \
                        _query_json_db(domain, dialogues_plus_acts[dia_id]['turns'][turn],
                                       dialogues_plus_acts[dia_id]['turns'][str(int(turn) + 1)]['utterance'],
                                       databases)
                    if len(db_state) > 0:
                        db_states[domain] = db_state
                    else:
                        db_states[domain] = None

            updated_context.append((dialogues_plus_acts[dia_id]['turns'][turn], db_states))

        if len(contexts) > 0:
            dialogue_contexts[dia_id] = {'services': dialogues_plus_acts[dia_id]['services'], 'contexts': contexts}
            num_unique_contexts += len(contexts)

    # Report
    print('Kept {} relevant dialogues, isolated {} unique contexts'.format(
        len(dialogue_contexts.keys()), num_unique_contexts))

    # Write the collected contexts to disc
    with open(out_path, 'w', encoding='utf8') as out_f:
        json.dump(dialogue_contexts, out_f, indent=3, sort_keys=True, ensure_ascii=False)
    print('Saved collected contexts to {}'.format(out_path))


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
