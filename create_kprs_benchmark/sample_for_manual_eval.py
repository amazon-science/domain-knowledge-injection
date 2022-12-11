import json
import random
import argparse

from util import read_jsonl


def sample_samples(jsonl_paths, num_samples, out_path):
    """ Randomly draws num_samples samples from the .jsonl dataset for manual inspection. """

    # Read-in samples from all sources
    all_samples = list()
    for jp in jsonl_paths:
        samples = read_jsonl(jp)
        for s in samples:
            s['domain'] = 'single' if 'single' in jp else 'multi'
        all_samples += samples

    # Shuffle samples
    random.shuffle(all_samples)

    # Write to separate file for eval
    with open(out_path, 'w', encoding='utf8') as out_f:
        for line_id, line in enumerate(all_samples[: num_samples]):
            if line_id > 0:
                out_f.write('\n')
            out_f.write(json.dumps(line))
            out_f.write('\n\n')

    print('Saved {} sampled samples to {}'.format(num_samples, out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_paths', type=str, nargs='+', required=True,
                        help='paths pointing to the datasets to sample from')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='path to the database directory')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path to which the samples samples should be written')
    args = parser.parse_args()

    sample_samples(args.jsonl_paths, args.num_samples, args.out_path)
