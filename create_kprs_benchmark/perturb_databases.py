import json
import random
import argparse

DOMAINS = ['restaurant', 'hotel', 'attraction', 'train']


def perturb_db(database_dir, out_dir, domain_to_perturb):
    """ Perturbs JSON databases by reassigning entity names across entries; if domain_to_perturb is specified,
    only perturbs the corresponding DB """

    # Read-in databases
    databases = dict()
    for domain in DOMAINS:
        db = '{}/{}_db.json'.format(database_dir, domain)
        with open(db, 'r', encoding='utf8') as in_f:
            databases[domain] = json.load(in_f)

    # Permute each database
    mappings = dict()
    new_databases = dict()
    for domain in databases.keys():

        # Only perturb the specified domain, if provided
        if domain_to_perturb is not None:
            if domain != domain_to_perturb:
                continue

        mappings[domain] = dict()
        new_databases[domain] = list()

        name_key = 'trainID' if domain == 'train' else 'name'
        names = [entry[name_key] for entry in databases[domain]]
        random.shuffle(names)
        names_taken = set()
        for entry in databases[domain]:
            name_id = 0
            try:
                while entry[name_key] == names[name_id] or names[name_id] in names_taken:
                    name_id += 1
            except IndexError:
                break  # catches duplicate entries in the train domain
            names_taken.add(names[name_id])
            mappings[domain][entry[name_key]] = names[name_id]
            new_entry = {key: val for key, val in entry.items()}
            new_entry[name_key] = names[name_id]
            new_databases[domain].append(new_entry)

    # Write permuted databases to file
    print('Writing permuted domains and their mappings to files ...')
    for domain in new_databases.keys():
        db_out_path = '{}/{}_db.json'.format(out_dir, domain)
        with open(db_out_path, 'w', encoding='utf8') as out_db:
            json.dump(new_databases[domain], out_db, indent=3, sort_keys=True, ensure_ascii=False)

        mapping_out_path = '{}/{}_db_mapping.json'.format(out_dir, domain)
        with open(mapping_out_path, 'w', encoding='utf8') as out_map:
            json.dump(mappings[domain], out_map, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir', type=str, required=True,
                        help='path to the database directory')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to the directory for the permuted databases')
    parser.add_argument('--domain_to_perturb', type=str, default=None,
                        help='path to the directory for the permuted databases')
    args = parser.parse_args()

    perturb_db(args.database_dir, args.out_dir, args.domain_to_perturb)
