# 1. Only consider responses that contain DB information
# 2. For the response, check DB query result
# 3. Identify which bits of the response communicate DB information in reference and whether all slots(?) are covered
# 4. Do the same for the system-generated response
# If response mentions entity name / ID, expand the set of permissible entities according to the DB info (similar to inform rate)
# If response provides any other information, look for exact match (requestables)
# NOTE: Contrary to previous evaluation methods, this evaluation is turn-based, instead of spanning the full dialogue

import string

import numpy as np

from word2number import w2n
from util import load_mwoz_databases_json


def _check_response_correctness(generated_response, true_response, context, databases):
    """ Checks if the model-generated dialogue response provides correct and relevant information """

    # Identify types and values of DB information present in the reference response
    ignore_system_acts = ['request', 'select']
    ignore_domains = ['booking', 'general']
    # NOTE: internet / parking are difficult to evaluate via string match and are, therefore, excluded
    ignore_slots = ['bookday', 'bookpeople', 'bookstay', 'booktime', 'none', 'internet', 'parking']

    # Go through the context and identify requested slots, and values provided by the system responses
    pending_requested_slots = dict()  # dict values denote if the information has already been provided
    for turn in context:
        # User turn
        if turn[0]['speaker'] == 'USER':
            for domain_info in turn[0]['frames']:
                for rq in domain_info['state']['requested_slots']:
                    request_domain, request_slot = rq.lower().split('-')  # e.g. rq = 'attraction-postcode'
                    if pending_requested_slots.get(request_domain, None) is None:
                        pending_requested_slots[request_domain] = dict()
                    pending_requested_slots[request_domain][request_slot] = 0
        else:
            # System turn
            for act in turn[0]['dialog_act'].keys():
                act_domain, act_type = act.lower().split('-')  # e.g. act = 'Hotel-Inform'
                if act_domain in pending_requested_slots.keys():
                    for info in turn[0]['dialog_act'][act]:
                        act_slot = info[0]
                        if act_slot in pending_requested_slots[act_domain].keys():
                            pending_requested_slots[act_domain][act_slot] = 1

    # Isolate mentions of entities and fetch their DB entries
    ref_entities = dict()
    for act in true_response['dialog_act'].keys():
        act_domain, act_type = act.lower().split('-')
        for slot in true_response['dialog_act'][act]:
            if slot[0] in ['name', 'trainid']:
                try:  # to account for name mismatches / misspellings etc.
                    ref_entities[slot[1].lower()] = databases[act_domain][slot[1].lower()]
                except KeyError:
                    continue
    # Create a second dictionary with all admissible entities according to the last DB call
    all_permissible_entities = dict()
    last_db_result = context[-1][-1]  # domain: DB call results

    # Isolate entity names from DB results; i.e. all entities that would be appropriate / correct to mention
    for domain in last_db_result.keys():
        if last_db_result[domain] is None:
            continue
        else:
            all_permissible_entities[domain] = dict()
        for entry in last_db_result[domain]:
            entity_id = 0
            if domain == 'restaurant':
                entity_id = 4
            if domain == 'hotel':
                entity_id = 7
            if domain == 'attraction':
                entity_id = 3
            if domain == 'train':
                entity_id = 0
            try:
                all_permissible_entities[domain][entry[entity_id]] = dict()
                for slot_type in databases[domain][entry[entity_id]].keys():
                    val = databases[domain][entry[entity_id]][slot_type]
                    if type(val) == str:
                        val = val.lower()
                    all_permissible_entities[domain][entry[entity_id]][slot_type] = val
            except KeyError:
                continue

    # Needed for the computation of false positive requestables
    # Collect all possible values per domain+slot
    all_possible_slot_values = \
        {domain: dict() for domain, values in all_permissible_entities.items() if values is not None}
    for domain in all_possible_slot_values.keys():
        for entity in databases[domain].keys():
            for slot_type in databases[domain][entity].keys():
                if slot_type not in all_possible_slot_values[domain].keys():
                    all_possible_slot_values[domain][slot_type] = list()
                val = databases[domain][entity][slot_type]
                if type(val) == str:
                    val = val.lower()
                if val not in all_possible_slot_values[domain][slot_type]:
                    all_possible_slot_values[domain][slot_type].append(val)

    # Collect all values per domain+slot that are supported by the dialogue context
    all_permissible_slot_values = \
        {domain: dict() for domain, values in all_permissible_entities.items() if values is not None}
    for domain in all_permissible_entities.keys():
        for entity in all_permissible_entities[domain].keys():
            for slot in all_permissible_entities[domain][entity].keys():
                if slot not in all_permissible_slot_values[domain].keys():
                    all_permissible_slot_values[domain][slot] = list()
                val = all_permissible_entities[domain][entity][slot]
                if type(val) == str:
                    val = val.lower()
                if val not in all_permissible_slot_values[domain][slot]:
                    all_permissible_slot_values[domain][slot].append(val)

    # Check which (if any) entities appear in the generated response
    all_entities = list()
    for domain in databases.keys():
        all_entities += list(databases[domain].keys())
    gen_entities = list()
    for ent in all_entities:
        if ent.lower() in generated_response.lower():
            gen_entities.append(ent)

    # Evaluate the generated response
    all_slots_filled = list()
    inform_slots_filled = list()
    request_slots_filled = list()
    inform_false_positives, request_false_positives = 0, 0
    entities_count = [0, 0]
    choices_count = [0, 0]
    requestables_count = [0, 0]
    no_offer_flags = [0, 0]
    sample_domains = list()
    for act in true_response['dialog_act'].keys():
        act_domain, act_type = act.strip().lower().split('-')

        # Ignore irrelevant information
        if (act_domain in ignore_domains) or (act_type in ignore_system_acts):
            continue
        else:
            sample_domains.append(act_domain)
            for slot in true_response['dialog_act'][act]:
                if slot[0] in ignore_slots:
                    continue
                if slot[1] in ["", "dontcare", "not mentioned", "don't care", "dont care", "do n't care", "?"]:
                    pass

                # Cover the NoOffer case: if response contains any entity names, count it as incorrect
                if act_type == 'nooffer':
                    no_offer_flags[1] = 1
                    if len(gen_entities) > 0:
                        all_slots_filled.append(0)
                    else:
                        all_slots_filled.append(1)
                        no_offer_flags[0] = 1

                # Check if the generated response mentions any of the permissible entities
                if slot[0] in ['name', 'trainid'] and slot[1] is not None:
                    try:
                        flag = False
                        for entity in all_permissible_entities[act_domain]:
                            if entity.lower() in generated_response.lower():
                                flag = True
                                break
                        if flag:
                            inform_slots_filled.append(1)
                            entities_count[0] += 1
                        else:
                            inform_slots_filled.append(0)
                        # Check for false positives
                        for entity in databases[act_domain].keys():
                            if entity.lower() in generated_response.lower() and \
                                    (entity not in all_permissible_entities[act_domain].keys() and
                                     entity.lower() not in all_permissible_entities[act_domain].keys()):
                                inform_false_positives += 1
                        entities_count[1] += 1
                    except KeyError:
                        return [None] * 11

                # Check if the generated response contains an accurate count of entities
                if slot[0] == 'choice':
                    choice_val = slot[1]
                    gen_tokens = \
                        [t.strip().strip(string.punctuation).lower() for t in generated_response.split(' ')]
                    flag = False
                    # Check for presence of the correct choice mention in the generated response
                    if choice_val.lower() in generated_response.lower():
                        flag = True
                    if not flag:
                        # Try to convert the reference choice value to an integer
                        try:
                            choice_val_num = str(w2n.word_to_num(choice_val.lower()))
                        except ValueError:
                            choice_val_num = choice_val
                        # Convert all numeral words to integers in the generated response
                        gen_tokens_num = list()
                        for tok in gen_tokens:
                            try:
                                gen_tokens_num.append(str(w2n.word_to_num(tok.lower())))
                            except ValueError:
                                gen_tokens_num.append(tok)
                        # Check if the generated response mentions the right value
                        if choice_val_num in gen_tokens or choice_val_num in gen_tokens_num or \
                                choice_val in gen_tokens_num:
                            flag = True
                    if flag:
                        all_slots_filled.append(1)
                        choices_count[0] += 1
                    else:
                        all_slots_filled.append(0)
                    choices_count[1] += 1

                # NOTE: slots requested in test dialogues:
                # {'ref', 'area', 'internet', 'type', 'trainid', 'price', 'postcode', 'address', 'parking',
                # 'duration', 'name', 'food', 'stars', 'phone', 'leaveat', 'arriveby', 'pricerange', 'entrancefee'}
                # excluded are name + trainid as they are covered by the inform rate, as well as internet / parking as
                # exact string matching doesn't work well for them

                # Requestables are conditioned on dialogue context and therefore must be an !exact match!
                # Still, this is a rough heuristic only, as values such as 'entrancefee' can be expressed in multiple,
                # equally valid ways

                if slot[0] in ['area', 'type', 'price', 'postcode', 'address', 'duration', 'food', 'stars', 'phone',
                               'leaveat', 'arriveby', 'pricerange', 'entrancefee']:

                    # Check if value was requested
                    is_requestable = False
                    if act_domain in pending_requested_slots.keys():
                        if slot[0] in pending_requested_slots[act_domain]:
                            if pending_requested_slots[act_domain][slot[0]] == 0:
                                is_requestable = True
                    flag = False

                    def _check_star_match(_flag, star_value):
                        """ Helper function to cover numerical and verbal star mentions """
                        if star_value.lower() in generated_response.lower():
                            _flag = True
                        if not _flag:
                            try:
                                star_value = str(w2n.word_to_num(star_value.lower()))
                            except ValueError:
                                pass
                            if star_value.lower() in generated_response.lower():
                                _flag = True
                        if not _flag:
                            sent_tokens = \
                                [t.strip().strip(string.punctuation).lower() for t in generated_response.split(' ')]
                            sent_tokens_num = list()
                            for _tok in sent_tokens:
                                try:
                                    sent_tokens_num.append(str(w2n.word_to_num(_tok.lower())))
                                except ValueError:
                                    sent_tokens_num.append(_tok)
                            if star_value.lower() in sent_tokens_num:
                                _flag = True
                        return _flag

                    if is_requestable:
                        try:
                            if slot[0] == 'stars':
                                flag = _check_star_match(flag, slot[1])
                                # Check for false positives
                                for val in all_possible_slot_values[act_domain]['stars']:
                                    if val != slot[1]:
                                        try:
                                            val_mod = w2n.word_to_num(val)
                                            if val_mod == slot[1]:
                                                continue
                                        except ValueError:
                                            pass
                                        request_false_positives += int(_check_star_match(False, val))

                            else:
                                if slot[1].lower() in generated_response.lower():
                                    flag = True
                                if not flag:
                                    for val in all_permissible_slot_values[act_domain][slot[0]]:
                                        if val.lower() in generated_response.lower():
                                            flag = True
                                            break
                                # Check for false positives
                                for val in all_possible_slot_values[act_domain][slot[0]]:
                                    is_substring = False
                                    for perm_val in all_permissible_slot_values[act_domain][slot[0]]:
                                        if val in perm_val:
                                            is_substring = True
                                            break
                                    if val.lower() in generated_response.lower() and \
                                            (val.lower() not in all_permissible_slot_values[act_domain][slot[0]]
                                             and not is_substring):
                                        request_false_positives += 1

                            if flag:
                                request_slots_filled.append(1)
                                pending_requested_slots[act_domain][slot[0]] = 1
                                requestables_count[0] += 1
                            else:
                                request_slots_filled.append(0)
                            requestables_count[1] += 1
                        except KeyError:
                            return [None] * 11

    # Also check whether a correct entity suggestion was generated when the reference did not contain any
    if entities_count[1] == 0:
        for act_domain in sample_domains:
            if act_domain in all_permissible_entities.keys():
                for entity in all_permissible_entities[act_domain]:
                    if entity.lower() in generated_response.lower():
                        entities_count[0] += 1
                        break

    # Determine inform label (i.e. mentioned at least as many entities as the reference response)
    if entities_count[1] > 0:
        inform_ratio = entities_count[0] / entities_count[1]
        if inform_ratio >= 1.:
            inform_label = True
        else:
            inform_label = False
    else:
        inform_label, inform_ratio = None, None

    # Also consider cases where entities were not mentioned in the reference
    if entities_count[1] == 0:
        if entities_count[0] > 0:
            inform_label_noref = True
        else:
            inform_label_noref = False
    else:
        inform_label_noref = None

    # Determine request label (i.e. mentioned at least as many requestables as the reference response)
    if requestables_count[1] > 0:
        request_ratio = requestables_count[0] / requestables_count[1]
        if request_ratio >= 1.:
            request_label = True
        else:
            request_label = False
    else:
        request_label, request_ratio = None, None

    # Determine choice label (i.e. mentioned the right number of choices)
    if choices_count[1] > 0:
        choice_ratio = choices_count[0] / choices_count[1]
        if choice_ratio >= 1.:
            choice_label = True
        else:
            choice_label = False
    else:
        choice_label = None

    # Determine NoOffer label (i.e. whether the generated response indicates no offer when required)
    # NOTE: Not used in the final evaluation, to be closer to previous work
    if no_offer_flags[1] > 0:
        if no_offer_flags[0] == 1:
            correct_no_offer = True
        else:
            correct_no_offer = False
    else:
        correct_no_offer = None

    return inform_slots_filled, inform_false_positives, request_slots_filled, request_false_positives, inform_label, \
        inform_ratio, request_label, request_ratio, choice_label, correct_no_offer, inform_label_noref


def evaluate_all_samples(samples, generated_responses, database_dir, requires_db_only=True):
    """ Evaluates the entirety of the test / dev corpus. """

    # Initialize containers
    inform_turn_labels = {'success': 0, 'success_noref': 0, 'failure': 0, 'ratios': list()}
    request_turn_labels = {'success': 0, 'failure': 0, 'ratios': list()}
    choice_labels = list()
    no_offer_labels = list()
    all_inform_mentions = {'true_pos': 0, 'false_pos': 0, 'false_neg': 0}
    all_request_mentions = {'true_pos': 0, 'false_pos': 0, 'false_neg': 0}

    # Read-in databases and restructure
    dbs = load_mwoz_databases_json(database_dir)
    new_dbs = {dom: dict() for dom in dbs.keys()}
    for domain in dbs.keys():
        for entry in dbs[domain]:
            if 'name' in entry.keys():
                new_dbs[domain][entry['name']] = {k.lower().replace(' ', ''): v for k, v in entry.items()}
            if 'trainID' in entry.keys():
                new_dbs[domain][entry['trainID']] = {k.lower().replace(' ', ''): v for k, v in entry.items()}
    dbs = new_dbs

    for dial_num, dial_id in enumerate(samples.keys()):
        for sample_id, sample in enumerate(samples[dial_id]['samples']):

            # Pick the appropriate generated response
            guid = '{}-{}'.format(dial_id, sample_id)
            if guid not in generated_responses.keys():
                continue

            # sample_num += 1
            context = sample['context']
            response = sample['response']

            if requires_db_only and not sample['requires_db']:
                continue  # only evaluate model performance on samples that require DB access
            # Also skip responses pertaining to booking as we are not interested in those
            skip_sample = False
            for act in response['dialog_act'].keys():
                act_domain, act_type = act.strip().lower().split('-')
                if act_domain in ['booking', 'general']:
                    skip_sample = True
                    break
            if skip_sample:
                continue

            # Analyze sample
            inform_slots_filled, inform_false_positives, request_slots_filled, request_false_positives, inform_label, \
                inform_ratio, request_label, req_ratio, choice_label, correct_no_offer, inform_label_noref = \
                _check_response_correctness(generated_responses[guid], response, context, dbs)

            # Skip invalid samples
            if inform_slots_filled is None:
                continue

            # Update trackers
            if inform_label is True:
                inform_turn_labels['success'] += 1
            if inform_label_noref is True:
                inform_turn_labels['success_noref'] += 1
            if inform_label is False:
                inform_turn_labels['failure'] += 1
            if inform_ratio is not None:
                inform_turn_labels['ratios'].append(inform_ratio)

            if request_label is True:
                request_turn_labels['success'] += 1
            if request_label is False:
                request_turn_labels['failure'] += 1
            if req_ratio is not None:
                request_turn_labels['ratios'].append(req_ratio)

            if choice_label is not None:
                choice_labels.append(choice_label)
            if correct_no_offer is not None:
                no_offer_labels.append(correct_no_offer)

            if len(inform_slots_filled) > 0:
                all_inform_mentions['true_pos'] += inform_slots_filled.count(1)
                all_inform_mentions['false_neg'] += inform_slots_filled.count(0)
            all_inform_mentions['false_pos'] += inform_false_positives

            if len(request_slots_filled) > 0:
                all_request_mentions['true_pos'] += request_slots_filled.count(1)
                all_request_mentions['false_neg'] += request_slots_filled.count(0)
            all_request_mentions['false_pos'] += request_false_positives

    def _compute_scores(true_pos, false_pos, false_neg):
        """ Helper function for computing slot precision / recall / F1 scores """
        precision = true_pos / float(true_pos + false_pos) if (true_pos + false_pos) != 0 else 0
        recall = true_pos / float(true_pos + false_neg) if (true_pos + false_neg) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return precision, recall, f1

    # Compute F1, separately for entities and requestables
    inform_precision, inform_recall, inform_f1 = _compute_scores(all_inform_mentions['true_pos'],
                                                                 all_inform_mentions['false_pos'],
                                                                 all_inform_mentions['false_neg'])
    request_precision, request_recall, request_f1 = _compute_scores(all_request_mentions['true_pos'],
                                                                    all_request_mentions['false_pos'],
                                                                    all_request_mentions['false_neg'])

    # Compute accuracies
    inform_acc = inform_turn_labels['success'] / (inform_turn_labels['success'] + inform_turn_labels['failure'])
    request_acc = request_turn_labels['success'] / (request_turn_labels['success'] + request_turn_labels['failure'])
    choice_acc = choice_labels.count(True) / len(choice_labels) if len(choice_labels) > 0 else 1.
    no_offer_acc = no_offer_labels.count(True) / len(no_offer_labels) if len(no_offer_labels) > 0 else 1.

    # Compute average ratios
    mean_inform_ratios = np.mean(inform_turn_labels['ratios'])
    mean_request_ratios = np.mean(request_turn_labels['ratios'])

    return inform_acc, request_acc, choice_acc, no_offer_acc, \
        inform_turn_labels['success_noref'], mean_inform_ratios, mean_request_ratios, \
        [inform_precision, inform_recall, inform_f1], [request_precision, request_recall, request_f1]
