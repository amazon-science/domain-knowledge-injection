# Annotates test samples for response generation with the set of permitted entities

import json
import argparse

from util import load_mwoz_databases_json


def annotate_with_entities(test_samples_path, test_generations_path, database_dir, out_path):
    """ Pairs contexts and targets with relevant database entites / requestables """

    # Read-in test samples
    with open(test_samples_path, 'r', encoding='utf8') as f:
        samples = json.load(f)

    # Read-in and format model generations
    with open(test_generations_path, 'r', encoding='utf8') as f:
        gen_lines = f.readlines()

    gen_segments = list()
    for line in gen_lines:
        input_seg = line.split("\"input\": ")[-1].split(", \"target\": ")[0]
        target_seq = line.split(", \"target\": ")[-1].split(", \"prediction\": ")[0]
        gen_segments.append({'input': input_seg[1: -1],
                             'target': target_seq[1: -1]})

    # Align (assumes all target responses to be unique)
    filtered_samples = list()
    num_with_sample = 0
    for gen_seg in gen_segments:
        matched = None
        for dia_id in samples.keys():
            for sample_id, sample in enumerate(samples[dia_id]['samples']):
                if gen_seg['target'].strip().lower()[8:] == sample['response']['utterance'].strip().lower():
                    matched = sample
                    num_with_sample += 1
                    break
            if matched:
                break
        if matched is None:
            print(gen_seg['target'].strip().lower()[8:])
        filtered_samples.append(matched)

    assert len(filtered_samples) == len(gen_segments), \
        'Couldn\'t assign samples to all generated responses. # samples: {}, # generations: {}'.format(
            len(filtered_samples), len(gen_segments))
    print("# targets with sample: {}".format(num_with_sample))
    print("# targets without sample: {}".format(len(filtered_samples) - num_with_sample))

    # Read-in databases and restructure
    databases = load_mwoz_databases_json(database_dir)
    new_databases = {dom: dict() for dom in databases.keys()}
    for domain in databases.keys():
        for entry in databases[domain]:
            if 'name' in entry.keys():
                new_databases[domain][entry['name']] = {k.lower().replace(' ', ''): v for k, v in entry.items()}
            if 'trainID' in entry.keys():
                new_databases[domain][entry['trainID']] = {k.lower().replace(' ', ''): v for k, v in entry.items()}
    databases = new_databases

    # For each sample, look up relevant DB info
    for sample_id, sample in enumerate(filtered_samples):
        all_permissible_entities = dict()
        if sample is not None:
            context = sample['context']
            last_db_result = context[-1][-1]

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
        gen_segments[sample_id]['supporting_db_info'] = all_permissible_entities

    # Write annotated input-output pairs to a new document
    print('Writing annotated targets to {}'.format(out_path))
    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(gen_segments, f, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_samples_path', type=str, required=True,
                        help='path to the files containing test samples')
    parser.add_argument('--test_generations_path', type=str, required=True,
                        help='path to a log containing model generations')
    parser.add_argument('--database_dir', type=str, required=True,
                        help='path to the MWOZ databases')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the destination file')
    args = parser.parse_args()

    annotate_with_entities(args.test_samples_path, args.test_generations_path, args.database_dir, args.out_path)



