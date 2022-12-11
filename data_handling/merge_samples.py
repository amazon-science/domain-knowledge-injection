import json
import argparse


def merge_samples(sample_paths):
    """ Merges single-domain and multi-domain samples into the same file. """

    # Collect samples from files
    all_samples = dict()
    for path in sample_paths:
        with open(path, 'r', encoding='utf8') as ip:
            table = json.load(ip)
            for dia_id in table.keys():
                if all_samples.get(dia_id, None) is None:
                    all_samples[dia_id] = table[dia_id]
                else:
                    for sample in table[dia_id]['samples']:
                        all_samples[dia_id]['samples'].append(sample)

    # Write to the combined file
    out_path = '/'.join(sample_paths[0].split('/')[:-1]) + '/merged_samples.json'
    with open(out_path, 'w', encoding='utf8') as op:
        json.dump(all_samples, op, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_to_samples', nargs='+', type=str, required=True,
                        help='path to the JSON file containing samples to be merged')
    args = parser.parse_args()

    merge_samples(args.paths_to_samples)
