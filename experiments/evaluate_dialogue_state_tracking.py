# External evaluation script to be run on logged DST model outputs
# (as the evaluation metric in the finetune_on_downstream_task.py script is faulty)

import argparse
from util import load_mwoz_databases_json


def eval_dst(log_file, database_dir):
    """ Parses the generation results and returns the joint goal accuracy score """

    # Read-in generations
    with open(log_file, 'r', encoding='utf8') as f:
        records = f.readlines()

    # Read in databases and check all possible slot value combinations
    databases = load_mwoz_databases_json(database_dir)
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

    # Select targets and model generations
    targets, generations = list(), list()
    for r in records:
        strings = r.split("\"target\":")[-1]
        targets.append(strings.split(", \"prediction\":".lower())[0])
        generations.append(strings.split(", \"prediction\":".lower())[-1])

    targets = [t.strip()[1:-1].strip() for t in targets]
    generations = [g.strip()[1:-2].strip() for g in generations]

    # Split targets into individual slot segments and check if they are in the generated response
    turns = {'num_correct': 0, 'num_incorrect': 0}
    slot_value_combos_tokens = {'valid': 0, 'invalid': 0}
    slot_value_combos_types = {'valid': list(), 'invalid': list()}
    for t_id, t in enumerate(targets):
        ref_slots = [seg.strip() for seg in t.split(', ')
                     if seg.strip().split(' ')[0].strip() in ['restaurant', 'hotel', 'attraction', 'train', 'mixed']]
        gen_slots = [seg.strip() for seg in generations[t_id].split(', ')
                     if seg.strip().split(' ')[0].strip() in ['restaurant', 'hotel', 'attraction', 'train', 'mixed']]
        slot_hits = [slot in generations[t_id] for slot in ref_slots]
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
        slot_value_combos_tokens['valid'] / (slot_value_combos_tokens['valid'] + slot_value_combos_tokens['invalid'])
    validity_ratio_types = len(slot_value_combos_types['valid']) / \
        (len(slot_value_combos_types['valid']) + len(slot_value_combos_types['invalid']))
    print('Joint Goal Accuracy: {}'.format(joint_goal_accuracy))
    print('# valid slot combos tokens: {}, # invalid slot combos: {}, validity ratio: {}'.format(
        slot_value_combos_tokens['valid'], slot_value_combos_tokens['invalid'], validity_ratio_tokens))
    print('# valid slot combos types: {}, # invalid slot combos: {}, validity ratio: {}'.format(
        len(slot_value_combos_types['valid']), len(slot_value_combos_types['invalid']), validity_ratio_types))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, required=True,
                        help='path to the log file to be evaluated')
    parser.add_argument('--database_dir', type=str, required=True,
                        help='path to the MWOZ databases')
    args = parser.parse_args()

    eval_dst(args.log_file, args.database_dir)



