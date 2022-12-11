import json
import argparse

ENTITY_KEYS = {'restaurant': 'name', 'hotel': 'name', 'attraction': 'name', 'train': 'trainid'}


def _collect_entity_mentions(samples_table, distractors):
    """ Identifies entities mentioned in a particular data split """

    mentioned_entities = set()

    # Iterate through samples and collect entity mentions
    for dia_id in samples_table.keys():
        for sample_id, sample in enumerate(samples_table[dia_id]['samples']):
            for turn in sample['context']:
                for dom in sample['sample_domains']:
                    if dom not in ENTITY_KEYS.keys():
                        continue
                    relevant_entities = distractors[dom.lower()][ENTITY_KEYS[dom.lower()]]
                    for ent in relevant_entities:
                        if ent.lower() in turn[0]['utterance'].lower():
                            mentioned_entities.add(ent)

            for dom in sample['sample_domains']:
                if dom not in ENTITY_KEYS.keys():
                    continue
                relevant_entities = distractors[dom.lower()][ENTITY_KEYS[dom.lower()]]
                for ent in relevant_entities:
                    if ent.lower() in sample['true_response']['utterance'].lower():
                        mentioned_entities.add(ent)

    return mentioned_entities


def filter_train(train_path, dev_path, test_path, distractors_path):
    """ Removes training samples that contain entities that are mentioned in test and dev sets. """

    # Read in data
    with open(train_path, 'r', encoding='utf8') as trp:
        train_table = json.load(trp)

    with open(dev_path, 'r', encoding='utf8') as dep:
        dev_table = json.load(dep)

    with open(test_path, 'r', encoding='utf8') as tep:
        test_table = json.load(tep)

    with open(distractors_path, 'r', encoding='utf8') as dip:
        distractors_table = json.load(dip)

    # Identify entities in test and dev
    dev_entities = _collect_entity_mentions(dev_table, distractors_table)
    test_entities = _collect_entity_mentions(test_table, distractors_table)
    dev_test_entities = dev_entities.union(test_entities)

    print('Collected entity mentions!')

    # Filter out training dialogues
    samples_seen = 0
    kept_samples, removed_samples = 0, 0
    filtered_training_dialogues = dict()

    # Iterate through training samples and only keep ones that mention entities that
    # do not occur in the dev / test splits
    for dia_id in train_table.keys():
        hit = False
        for sample_id, sample in enumerate(train_table[dia_id]['samples']):
            samples_seen += 1
            if samples_seen % 100 == 0:
                print('Checked {} samples'.format(samples_seen))
            for turn in sample['context']:
                for ent in dev_test_entities:
                    if ent.lower() in turn[0]['utterance'].lower():
                        hit = True
                        break
                    if hit:
                        break
                if hit:
                    break

            if not hit:
                for ent in dev_test_entities:
                    if ent.lower() in sample['true_response']['utterance'].lower():
                        hit = True
                        break
            if not hit:
                if filtered_training_dialogues.get(dia_id, None) is None:
                    filtered_training_dialogues[dia_id] = {k: v for k, v in train_table[dia_id].items()}
                    filtered_training_dialogues[dia_id]['samples'] = list()
                filtered_training_dialogues[dia_id]['samples'].append(sample)
                kept_samples += 1
            else:
                removed_samples += 1

    print('Kept {} samples, dropped {} samples'.format(kept_samples, removed_samples))

    filtered_path = train_path[:-5] + '_filtered.json'
    with open(filtered_path, 'w', encoding='utf8') as out_f:
        json.dump(filtered_training_dialogues, out_f, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                        help='path to the training data')
    parser.add_argument('--dev_path', type=str, required=True,
                        help='path to the dev data')
    parser.add_argument('--test_path', type=str, required=True,
                        help='path to the test data')
    parser.add_argument('--distractors_path', type=str, required=True,
                        help='path to the distractors')
    args = parser.parse_args()

    filter_train(args.train_path, args.dev_path, args.test_path, args.distractors_path)
