import json
import argparse


def combine_training_data(dialogue_file_paths, out_path):
    """ Helper function to combine all MultiWoZ training dialogues in a single file. """

    # Combine files
    all_dialogues = list()
    for path in dialogue_file_paths:
        with open(path, 'r', encoding='utf8') as dia_in:
            dialogues = json.load(dia_in)
            all_dialogues += dialogues

    # Write to file
    with open(out_path, 'w', encoding='utf8') as dia_out:
        json.dump(all_dialogues, dia_out, indent=3, sort_keys=True, ensure_ascii=False)

    # Report
    print('Wrote {} dialogues to {}'.format(len(all_dialogues), out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogue_file_paths', type=str, nargs='+', required=True,
                        help='list of paths to the files containing dialogues')
    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the destination file')
    args = parser.parse_args()

    combine_training_data(args.dialogue_file_paths, args.out_path)
