import os
import re
import sys
import csv
import json
import glob
import shutil
import logging
import sqlite3

from io import open

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, mask_start_id=None, mask_end_id=None):
        """ Constructs an InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.mask_start_id = mask_start_id
        self.mask_end_id = mask_end_id


class InputExampleKI(object):
    """ A single training / dev / test example for the knowledge injection step. """
    def __init__(self, guid, slot, masked_fact, target, slot_seq, entity):
        self.guid = guid
        self.slot = slot
        self.masked_fact = masked_fact
        self.target = target
        self.slot_seq = slot_seq
        self.entity = entity


class InputExampleDS(object):
    """ A single training / dev / test example for the knowledge injection step. """
    def __init__(self, guid, domain, input_seq, target_seq, requires_db=None, contents=None,
                 slot=None, slot_seq=None, entity=None):
        self.guid = guid
        self.domain = domain
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.requires_db = requires_db
        self.contents = contents  # DST / DB information (depends on the task)

        self.slot = slot
        self.slot_seq = slot_seq
        self.entity = entity


class InputFeatures(object):
    """ A single set of features of data. """
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputFeaturesKI(object):
    """ A single set of features of data used during the knowledge-injection step with the MLM objective """

    def __init__(self, input_ids, input_mask, segment_ids, target_ids, target_mask, slot, slot_seq, entity):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.target_ids = target_ids
        self.target_mask = target_mask
        self.slot = slot
        self.slot_seq = slot_seq  # slot name as mentioned in the verbalized fact
        self.entity = entity  # entity name as mentioned in the verbalized fact


class InputFeaturesDS(object):
    """ A single set of features of data used during the knowledge-injection step with the MLM objective """

    def __init__(self, guid, domain, input_ids, input_mask, segment_ids, target_ids, target_mask, need_db, contents,
                 slot, slot_seq, entity):
        self.guid = guid
        self.domain = domain
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.target_ids = target_ids
        self.target_mask = target_mask
        self.need_db = need_db
        self.contents = contents  # DST / DB information (depends on the task)

        self.slot = slot
        self.slot_seq = slot_seq
        self.entity = entity


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, tokenizer, args=None):
        self.tokenizer = tokenizer
        self.args = args

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)


class TupleAtomicProcessor(DataProcessor):
    """ Processes database facts verbalized as relational tuples and atomic facts """

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        return self._create_examples(self._read_json(os.path.join(data_dir, 'train.json')))

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        return self._create_examples(self._read_json(os.path.join(data_dir, 'train.json')), evaluate=True)

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        return None

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Reads a tab separated value file. """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """ Reads a JSON file """
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)

    def _create_examples(self, table, evaluate=False):
        """ Creates examples for the training and dev sets. """
        examples = dict()
        for domain in table.keys():
            examples[domain] = list()
            for (i, line) in enumerate(table[domain]):
                # Read-in sample
                guid = i
                slot = line['slot']
                slot_seq = line['slot_seq']
                entity = line['entity']
                fact = line['fact']
                mask_positions = line['slot_value_ws_token_positions']

                # Mask model input
                if len(mask_positions) == 1:
                    masked_fact = \
                        ' '.join([tok if tok_idx != mask_positions[0] else self.tokenizer.mask_token for
                                  tok_idx, tok in enumerate(fact.split(' '))])
                    masked_info = fact.split(' ')[mask_positions[0]]
                else:
                    masked_fact = \
                        ' '.join([tok if (tok_idx < mask_positions[0] or tok_idx > mask_positions[-1]) else
                                  self.tokenizer.mask_token for tok_idx, tok in enumerate(fact.split(' '))])
                    # Collapse masks
                    masked_fact = masked_fact.replace(' '.join([self.tokenizer.mask_token] * len(mask_positions)),
                                                      self.tokenizer.mask_token)
                    masked_info = ' '.join([tok for tok_idx, tok in enumerate(fact.split(' ')) if
                                            (mask_positions[0] <= tok_idx <= mask_positions[-1])])

                if fact.split(' ')[mask_positions[-1]].endswith('.'):
                    masked_fact = masked_fact.replace(self.tokenizer.mask_token, self.tokenizer.mask_token + '.')
                if masked_info.endswith('.'):
                    masked_info = masked_info[:-1]

                # Build dataset sample
                if not evaluate:
                    examples[domain].append(
                        InputExampleKI(guid=guid,
                                       slot=slot,
                                       masked_fact=masked_fact,
                                       target=fact,
                                       slot_seq=slot_seq,
                                       entity=entity))
                else:
                    examples[domain].append(
                        InputExampleKI(guid=guid,
                                       slot=slot,
                                       masked_fact=masked_fact,
                                       target=masked_info.strip(),
                                       slot_seq=slot_seq,
                                       entity=entity))

        return examples


class CompositeProcessor(DataProcessor):
    """ Processes database facts verbalized as compositional facts """

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        return self._create_examples(self._read_json(os.path.join(data_dir, 'train.json')))

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        return self._create_examples(self._read_json(os.path.join(data_dir, 'train.json')), evaluate=True)

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        return None

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Reads a tab separated value file. """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """ Reads a JSON file """
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)

    def _create_examples(self, table, evaluate=False):
        """ Creates examples for the training set. """
        examples = dict()
        for domain in table.keys():
            examples[domain] = list()
            for (i, line) in enumerate(table[domain]):
                # Read-in sample
                guid = i
                fact = line['fact']
                slot_seqs = line['slot_seq']
                entity = line['entity']
                mask_positions_dict = line['slot_value_ws_token_positions']

                for slot in mask_positions_dict.keys():
                    # Mask model input
                    mask_positions = mask_positions_dict[slot]
                    slot_seq = slot_seqs[slot]
                    if len(mask_positions) == 1:
                        masked_fact = \
                            ' '.join([tok if tok_idx != mask_positions[0] else self.tokenizer.mask_token for
                                      tok_idx, tok in enumerate(fact.split(' '))])
                        masked_info = fact.split(' ')[mask_positions[0]]
                    else:
                        masked_fact = \
                            ' '.join([tok if (tok_idx < mask_positions[0] or tok_idx > mask_positions[-1]) else
                                      self.tokenizer.mask_token for tok_idx, tok in enumerate(fact.split(' '))])
                        # Collapse masks
                        masked_fact = masked_fact.replace(' '.join([self.tokenizer.mask_token] * len(mask_positions)),
                                                          self.tokenizer.mask_token)
                        masked_info = ' '.join([tok for tok_idx, tok in enumerate(fact.split(' ')) if
                                                (mask_positions[0] <= tok_idx <= mask_positions[-1])])

                    if fact.split(' ')[mask_positions[-1]].endswith('.'):
                        masked_fact = masked_fact.replace(self.tokenizer.mask_token, self.tokenizer.mask_token + '.')
                    if masked_info.endswith('.'):
                        masked_info = masked_info[:-1]

                    # Build dataset sample
                    if not evaluate:
                        examples[domain].append(
                            InputExampleKI(guid=guid,
                                           slot=slot,
                                           masked_fact=masked_fact,
                                           target=fact,
                                           slot_seq=slot_seq,
                                           entity=entity))
                    else:
                        examples[domain].append(
                            InputExampleKI(guid=guid,
                                           slot=slot,
                                           masked_fact=masked_fact,
                                           target=masked_info.strip(),
                                           slot_seq=slot_seq,
                                           entity=entity))

        return examples


class AtomicCompositeProcessor(DataProcessor):
    """ Processes database facts verbalized as compositional facts """

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        comp_data_dir = '/'.join(data_dir.split('/')[:-1] + ['mwoz_composite_facts'])
        atomic_samples = self._create_atomic_examples(self._read_json(os.path.join(data_dir, 'train.json')))
        composite_samples = self._create_composite_examples(self._read_json(os.path.join(comp_data_dir, 'train.json')))
        mixed_samples = {domain: atomic_samples[domain] + composite_samples[domain] for domain in atomic_samples.keys()}
        return mixed_samples

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        comp_data_dir = '/'.join(data_dir.split('/')[:-1] + ['mwoz_composite_facts'])
        atomic_samples = \
            self._create_atomic_examples(self._read_json(os.path.join(data_dir, 'train.json')), evaluate=True)
        composite_samples = \
            self._create_composite_examples(self._read_json(os.path.join(comp_data_dir, 'train.json')), evaluate=True)
        mixed_samples = {domain: atomic_samples[domain] + composite_samples[domain] for domain in atomic_samples.keys()}
        return mixed_samples

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        return None

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Reads a tab separated value file. """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """ Reads a JSON file """
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)

    def _create_atomic_examples(self, table, evaluate=False):
        """ Creates examples for the training and dev sets. """
        examples = dict()
        for domain in table.keys():
            examples[domain] = list()
            for (i, line) in enumerate(table[domain]):
                # Read-in sample
                guid = i
                slot = line['slot']
                slot_seq = line['slot_seq']
                entity = line['entity']
                fact = line['fact']
                mask_positions = line['slot_value_ws_token_positions']

                # Mask model input
                if len(mask_positions) == 1:
                    masked_fact = \
                        ' '.join([tok if tok_idx != mask_positions[0] else self.tokenizer.mask_token for
                                  tok_idx, tok in enumerate(fact.split(' '))])
                    masked_info = fact.split(' ')[mask_positions[0]]
                else:
                    masked_fact = \
                        ' '.join([tok if (tok_idx < mask_positions[0] or tok_idx > mask_positions[-1]) else
                                  self.tokenizer.mask_token for tok_idx, tok in enumerate(fact.split(' '))])
                    # Collapse masks
                    masked_fact = masked_fact.replace(' '.join([self.tokenizer.mask_token] * len(mask_positions)),
                                                      self.tokenizer.mask_token)
                    masked_info = ' '.join([tok for tok_idx, tok in enumerate(fact.split(' ')) if
                                            (mask_positions[0] <= tok_idx <= mask_positions[-1])])

                if fact.split(' ')[mask_positions[-1]].endswith('.'):
                    masked_fact = masked_fact.replace(self.tokenizer.mask_token, self.tokenizer.mask_token + '.')
                if masked_info.endswith('.'):
                    masked_info = masked_info[:-1]

                # Build dataset sample
                if not evaluate:
                    examples[domain].append(
                        InputExampleKI(guid=guid,
                                       slot=slot,
                                       masked_fact=masked_fact,
                                       target=fact,
                                       slot_seq=slot_seq,
                                       entity=entity))
                else:
                    examples[domain].append(
                        InputExampleKI(guid=guid,
                                       slot=slot,
                                       masked_fact=masked_fact,
                                       target=masked_info.strip(),
                                       slot_seq=slot_seq,
                                       entity=entity))

        return examples

    def _create_composite_examples(self, table, evaluate=False):
        """ Creates examples for the training set. """
        examples = dict()
        for domain in table.keys():
            examples[domain] = list()
            for (i, line) in enumerate(table[domain]):
                # Read-in sample
                guid = i
                fact = line['fact']
                slot_seqs = line['slot_seq']
                entity = line['entity']
                mask_positions_dict = line['slot_value_ws_token_positions']

                for slot in mask_positions_dict.keys():
                    # Mask model input
                    mask_positions = mask_positions_dict[slot]
                    slot_seq = slot_seqs[slot]
                    if len(mask_positions) == 1:
                        masked_fact = \
                            ' '.join([tok if tok_idx != mask_positions[0] else self.tokenizer.mask_token for
                                      tok_idx, tok in enumerate(fact.split(' '))])
                        masked_info = fact.split(' ')[mask_positions[0]]
                    else:
                        masked_fact = \
                            ' '.join([tok if (tok_idx < mask_positions[0] or tok_idx > mask_positions[-1]) else
                                      self.tokenizer.mask_token for tok_idx, tok in enumerate(fact.split(' '))])
                        # Collapse masks
                        masked_fact = masked_fact.replace(' '.join([self.tokenizer.mask_token] * len(mask_positions)),
                                                          self.tokenizer.mask_token)
                        masked_info = ' '.join([tok for tok_idx, tok in enumerate(fact.split(' ')) if
                                                (mask_positions[0] <= tok_idx <= mask_positions[-1])])

                    if fact.split(' ')[mask_positions[-1]].endswith('.'):
                        masked_fact = masked_fact.replace(self.tokenizer.mask_token, self.tokenizer.mask_token + '.')
                    if masked_info.endswith('.'):
                        masked_info = masked_info[:-1]

                    # Build dataset sample
                    if not evaluate:
                        examples[domain].append(
                            InputExampleKI(guid=guid,
                                           slot=slot,
                                           masked_fact=masked_fact,
                                           target=fact,
                                           slot_seq=slot_seq,
                                           entity=entity))
                    else:
                        examples[domain].append(
                            InputExampleKI(guid=guid,
                                           slot=slot,
                                           masked_fact=masked_fact,
                                           target=masked_info.strip(),
                                           slot_seq=slot_seq,
                                           entity=entity))

        return examples


class ResponseSelectionProcessor(DataProcessor):
    """ Prepares response generation samples """

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        samples_path = 'train/merged_samples.json' if len(self.args.active_domains) > 1 else \
            'train/single_domain_samples.json'
        return self._create_examples(self._read_json(os.path.join(data_dir, samples_path)))

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        samples_path = 'dev/merged_samples.json' if len(self.args.active_domains) > 1 else \
            'dev/single_domain_samples.json'
        return self._create_examples(self._read_json(os.path.join(data_dir, samples_path)))

    def get_test_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the test set. """
        raise NotImplementedError

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        return None

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Reads a tab separated value file. """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """ Reads a JSON file """
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)

    def _create_examples(self, table):
        """ Creates examples for the training split (KPRS evaluation is performed in a separate script). """
        examples = dict()

        for dia_id in table.keys():
            for sample_id, sample in enumerate(table[dia_id]['samples']):

                guid = '{}-{}'.format(dia_id, sample_id)

                # Skip samples not belonging to any domain
                if len(sample['sample_domains']) == 0:
                    continue

                # Construct model input and generation target
                context = list()
                for turn in sample['context']:
                    context.append('{}: {}'.format(turn[0]['speaker'], turn[0]['utterance']))
                input_seq = ' '.join(context)
                target_seq = '{}: {}'.format(sample['true_response']['speaker'], sample['true_response']['utterance'])

                # Determine which adapters will be needed
                domain_key = sample['sample_domains'][0] if len(sample['sample_domains']) == 1 else \
                    ','.join(sorted(sample['sample_domains']))

                if examples.get(domain_key, None) is None:
                    examples[domain_key] = list()
                examples[domain_key].append(InputExampleDS(guid=guid,
                                                           domain=domain_key,
                                                           input_seq=input_seq,
                                                           target_seq=target_seq,
                                                           requires_db=None))

        return examples


class ResponseGenerationProcessor(DataProcessor):
    """ Prepares response generation samples """

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        samples_path = 'train/merged_samples.json' if len(self.args.active_domains) > 1 else \
            'train/single_domain_samples.json'
        return self._create_examples(self._read_json(os.path.join(data_dir, samples_path)))

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        samples_path = 'dev/merged_samples.json' if len(self.args.active_domains) > 1 else \
            'dev/single_domain_samples.json'
        return self._create_examples(self._read_json(os.path.join(data_dir, samples_path)))

    def get_test_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the test set. """
        samples_path = 'test/merged_samples.json' if len(self.args.active_domains) > 1 else \
            'test/single_domain_samples.json'
        return self._create_examples(self._read_json(os.path.join(data_dir, samples_path)))

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        return None

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Reads a tab separated value file. """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """ Reads a JSON file """
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)

    def _create_examples(self, table):
        """ Creates examples for the training set. """
        examples = dict()
        for dia_id in table.keys():
            for sample_id, sample in enumerate(table[dia_id]['samples']):

                guid = '{}-{}'.format(dia_id, sample_id)

                # Skip samples not belonging to any domain
                if len(sample['sample_domains']) == 0:
                    continue

                # Construct model input and generation target
                context = list()
                for turn in sample['context']:
                    context.append('{}: {}'.format(turn[0]['speaker'], turn[0]['utterance']))
                input_seq = ' '.join(context)
                target_seq = '{}: {}'.format(sample['response']['speaker'], sample['response']['utterance'])

                # Determine which adapters will be needed
                domain_key = sample['sample_domains'][0] if len(sample['sample_domains']) == 1 else \
                    ','.join(sorted(sample['sample_domains']))
                if examples.get(domain_key, None) is None:
                    examples[domain_key] = list()
                examples[domain_key].append(InputExampleDS(guid=guid,
                                                           domain=domain_key,
                                                           input_seq=input_seq,
                                                           target_seq=target_seq,
                                                           contents=sample['response']['required_db_info'],
                                                           requires_db=sample['requires_db']))

        return examples


class StateTrackingProcessor(DataProcessor):
    """ Prepares dialogue state tracking samples """

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        samples_path = 'train/merged_samples.json' if len(self.args.active_domains) > 1 else \
            'train/single_domain_samples.json'
        return self._create_examples(self._read_json(os.path.join(data_dir, samples_path)))

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        samples_path = 'dev/merged_samples.json' if len(self.args.active_domains) > 1 else \
            'dev/single_domain_samples.json'
        return self._create_examples(self._read_json(os.path.join(data_dir, samples_path)))

    def get_test_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the test set. """
        samples_path = 'test/merged_samples.json' if len(self.args.active_domains) > 1 else \
            'test/single_domain_samples.json'
        return self._create_examples(self._read_json(os.path.join(data_dir, samples_path)))

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        return None

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Reads a tab separated value file. """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """ Reads a JSON file """
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)

    def _create_examples(self, table):
        """ Creates examples for the different splits. """
        examples = dict()
        for dia_id in table.keys():
            for sample_id, sample in enumerate(table[dia_id]['samples']):

                guid = '{}-{}'.format(dia_id, sample_id)

                # Skip samples not belonging to any domain
                if len(sample['sample_domains']) == 0:
                    continue

                # Construct model input and generation target
                context = list()
                for turn in sample['context']:
                    context.append('{}: {}'.format(turn[0]['speaker'], turn[0]['utterance']))
                input_seq = ' '.join(context)

                # Assemble dialogue state
                domains = sorted(list(sample['dialogue_state'].keys()))
                target_seq = ' '.join([sample['dialogue_state'][dom] for dom in domains])
                dst_contents = sample['dst_contents']

                # Determine which adapters will be needed
                domain_key = sample['sample_domains'][0] if len(sample['sample_domains']) == 1 else \
                    ','.join(sorted(sample['sample_domains']))
                if examples.get(domain_key, None) is None:
                    examples[domain_key] = list()
                examples[domain_key].append(InputExampleDS(guid=guid,
                                                           domain=domain_key,
                                                           input_seq=input_seq,
                                                           target_seq=target_seq,
                                                           contents=dst_contents,
                                                           requires_db=sample['requires_db']))

        return examples


DS_PROCESSORS = {
    'response_selection': ResponseSelectionProcessor,
    'response_generation': ResponseGenerationProcessor,
    'state_tracking': StateTrackingProcessor}

PROCESSORS = {
    'relational_tuples': TupleAtomicProcessor,
    'atomic_facts': TupleAtomicProcessor,
    'composite_facts': CompositeProcessor,
    'atomic_and_composite_facts': AtomicCompositeProcessor
}

# ======================================================================================================================


def convert_examples_to_features_ki(examples,
                                    max_seq_length,
                                    tokenizer,
                                    pad_on_left=False,
                                    pad_token_segment_id=0,
                                    sequence_a_segment_id=0,
                                    sequence_b_segment_id=1,
                                    mask_padding_with_zero=True,
                                    model_name='bart'):
    """ Converts text samples into features for use with the pre-trained language model (focusing on BART, for now) """

    # Save features per domain
    features = dict()

    for domain in examples.keys():
        features[domain] = list()

        # Initialize caches
        input_cache, target_cache, slot_cache, slot_seq_cache, entity_cache = list(), list(), list(), list(), list()

        # Iterate over samples
        for (ex_index, example) in enumerate(examples[domain]):
            if ex_index % 1000 == 0:
                logger.info('Writing example {} / {} for the {} domain'.format(ex_index, len(examples[domain]), domain))

            # Add special tokens
            masked_fact = tokenizer.eos_token + ' ' + example.masked_fact + tokenizer.eos_token
            target = tokenizer.eos_token + ' ' + example.target + tokenizer.eos_token
            # Tokenize the sample string (it's a single 'sentence' for tuples / atomic facts and a paragraph for
            # composite facts; the paragraph is treated as a single sentence - not sure if splitting makes sense)
            masked_tokens = tokenizer.tokenize(masked_fact)
            target_tokens = tokenizer.tokenize(target)

            # Convert inputs and targets to IDs
            input_cache.append([tokenizer.convert_tokens_to_ids(masked_tokens), masked_tokens])
            target_cache.append([tokenizer.convert_tokens_to_ids(target_tokens), target_tokens])
            slot_cache.append(example.slot)
            slot_seq_cache.append(example.slot_seq)
            entity_cache.append(example.entity)

        # Determine maximum sequence length for padding
        max_input_len, max_target_len, max_slot_len = max_seq_length, max_seq_length, max_seq_length

        # Make masks and pad inputs / labels
        for iid, inputs in enumerate(input_cache):
            example = examples[domain][iid]
            # Unpack
            input_ids, input_tokens = inputs
            segment_ids = [sequence_a_segment_id] * len(input_ids)
            target_ids, target_tokens = target_cache[iid]
            slot, slot_seq, entity = slot_cache[iid], slot_seq_cache[iid], entity_cache[iid]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length
            input_padding_length = max_input_len - len(input_ids)
            if pad_on_left:
                input_ids = ([tokenizer.pad_token_id] * input_padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * input_padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * input_padding_length) + segment_ids
            else:
                input_ids = input_ids + ([tokenizer.pad_token_id] * input_padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * input_padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * input_padding_length)

            # Mask out padding positions, so that they don't contribute to the loss calculation
            target_padding_length = max_target_len - len(target_ids)
            target_mask = [1 if mask_padding_with_zero else 0] * len(target_ids)
            if pad_on_left:
                target_ids = ([-100] * target_padding_length) + target_ids
                target_mask = ([0 if mask_padding_with_zero else 1] * target_padding_length) + target_mask
            else:
                target_ids = target_ids + ([-100] * target_padding_length)
                target_mask = target_mask + ([0 if mask_padding_with_zero else 1] * target_padding_length)

            try:
                assert len(input_ids) == len(input_mask) == max_input_len
                assert len(target_ids) == len(target_mask) == max_target_len
            except AssertionError:
                logging.info(input_ids, len(input_ids), len(input_mask))
                logging.info(target_ids, len(target_ids), len(target_mask))
                raise AssertionError

            if iid < 5:
                logger.info('*** Example ***')
                logger.info('guid: %s' % example.guid)
                logger.info('input_tokens: %s' % ' '.join(input_tokens))
                logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
                logger.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
                logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
                logger.info('target_tokens: %s' % ' '.join(target_tokens))
                logger.info('target_ids: %s' % ' '.join([str(x) for x in target_ids]))
                logger.info('target_mask: %s' % ' '.join([str(x) for x in target_mask]))
                logger.info('slot: %s' % slot)
                logger.info('slot in sequence: %s' % slot_seq)
                logger.info('fact entity: %s' % entity)

            features[domain].append(
                InputFeaturesKI(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                target_ids=target_ids,
                                target_mask=target_mask,
                                slot=slot,
                                slot_seq=slot_seq,
                                entity=entity))

    return features


def convert_examples_to_features_ds(examples,
                                    max_seq_length,
                                    tokenizer,
                                    pad_on_left=False,
                                    pad_token_segment_id=0,
                                    sequence_a_segment_id=0,
                                    sequence_b_segment_id=1,
                                    mask_padding_with_zero=True,
                                    model_name='bart',
                                    mix_samples=False,
                                    hard_max_len=False):
    """ Converts text samples into features for use with the pre-trained language model (focusing on BART, for now) """

    if mix_samples:
        # Collapse all examples into the 'mixed' domain for multi-domain training
        new_examples = {'mixed': list()}
        for key in examples.keys():
            new_examples['mixed'] += examples[key]
        examples = new_examples

    # Save features per domain
    features = dict()

    for domain in examples.keys():
        features[domain] = list()

        # Initialize caches
        input_cache, target_cache, requires_db_cache, domain_cache, contents_cache = \
            list(), list(), list(), list(), list()
        slot_cache, slot_seq_cache, entity_cache = list(), list(), list()

        # Iterate over samples
        for (ex_index, example) in enumerate(examples[domain]):
            if ex_index % 1000 == 0:
                logger.info('Writing example {} / {} for the {} domain'.format(ex_index, len(examples[domain]), domain))

            # Add special tokens
            input_seq = tokenizer.eos_token + ' ' + example.input_seq + tokenizer.eos_token
            target_seq = tokenizer.eos_token + ' ' + example.target_seq + tokenizer.eos_token
            # Tokenize the sample string (it's a single 'sentence' for tuples / atomic facts and a paragraph for
            # composite facts; the paragraph is treated as a single sentence - not sure if splitting makes sense)
            input_tokens = tokenizer.tokenize(input_seq)
            target_tokens = tokenizer.tokenize(target_seq)

            # Truncate the input (from the left, retaining most recent turns) if necessary
            if len(input_tokens) > max_seq_length:
                input_tokens = [tokenizer.eos_token] + input_tokens[-(max_seq_length - 1):]

            # Convert inputs and targets to IDs
            input_cache.append([tokenizer.convert_tokens_to_ids(input_tokens), input_tokens])
            target_cache.append([tokenizer.convert_tokens_to_ids(target_tokens), target_tokens])

            requires_db_cache.append(example.requires_db)
            domain_cache.append(example.domain)
            contents_cache.append(example.contents)

            slot_cache.append(example.slot)
            slot_seq_cache.append(example.slot_seq)
            entity_cache.append(example.entity)

        # Determine maximum sequence length for padding
        max_input_len, max_target_len, max_slot_len = max_seq_length, max_seq_length, max_seq_length

        # Note: Deactivated for better memory management
        if max_seq_length > 0 and not hard_max_len:
            max_input_len = max([len(s[0]) for s in input_cache])
            max_target_len = max([len(s[0]) for s in target_cache])

        # Make masks and pad inputs / labels
        for iid, inputs in enumerate(input_cache):
            example = examples[domain][iid]
            # Unpack
            input_ids, input_tokens = inputs
            segment_ids = [sequence_a_segment_id] * len(input_ids)
            target_ids, target_tokens = target_cache[iid]
            requires_db, sample_domain, contents = requires_db_cache[iid], domain_cache[iid], contents_cache[iid]
            slot, slot_seq, entity = slot_cache[iid], slot_seq_cache[iid], entity_cache[iid]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length
            input_padding_length = max_input_len - len(input_ids)
            if pad_on_left:
                input_ids = ([tokenizer.pad_token_id] * input_padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * input_padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * input_padding_length) + segment_ids
            else:
                input_ids = input_ids + ([tokenizer.pad_token_id] * input_padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * input_padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * input_padding_length)

            # Mask out padding positions, so that they don't contribute to the loss calculation
            target_padding_length = max_target_len - len(target_ids)
            target_mask = [1 if mask_padding_with_zero else 0] * len(target_ids)
            if pad_on_left:
                target_ids = ([-100] * target_padding_length) + target_ids
                target_mask = ([0 if mask_padding_with_zero else 1] * target_padding_length) + target_mask
            else:
                target_ids = target_ids + ([-100] * target_padding_length)
                target_mask = target_mask + ([0 if mask_padding_with_zero else 1] * target_padding_length)

            try:
                assert len(input_ids) == len(input_mask) == max_input_len
                assert len(target_ids) == len(target_mask) == max_target_len
            except AssertionError:
                logging.info(input_ids, len(input_ids), len(input_mask))
                logging.info(target_ids, len(target_ids), len(target_mask))
                raise AssertionError

            if iid < 5:
                logger.info('*** Example ***')
                logger.info('guid: %s' % example.guid)
                logger.info('input_tokens: %s' % ' '.join(input_tokens))
                logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
                logger.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
                logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
                logger.info('target_tokens: %s' % ' '.join(target_tokens))
                logger.info('target_ids: %s' % ' '.join([str(x) for x in target_ids]))
                logger.info('target_mask: %s' % ' '.join([str(x) for x in target_mask]))
                logger.info('requires_db: %s' % requires_db)
                logger.info('domain: %s' % domain)
                logger.info('contents: {}'.format(contents))

            features[domain].append(
                InputFeaturesDS(guid=example.guid,
                                domain=sample_domain,
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                target_ids=target_ids,
                                target_mask=target_mask,
                                need_db=requires_db,
                                contents=contents,
                                slot=slot,
                                slot_seq=slot_seq,
                                entity=entity))

    return features


# ======================================================================================================================

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    """ Keep a maximum of args.save_total_limit checkpoints (adopted from the SC101 scripts). """
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = list()
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info('Deleting older checkpoint [{}] due to args.save_total_limit'.format(checkpoint))
        shutil.rmtree(checkpoint)

# ======================================================================================================================
# Duplicate functions for data creation

def read_jsonl(file_path):
    """ Reads a .jsonl file. """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_mwoz_databases(database_dir):
    """ Helper function for loading the MultiWOZ databases (adapted from the MultiWOZ codebase) """
    # 'police' and 'taxi' domains are excluded due to simplicity, 'hotel' domain appears to be badly formatted?
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'bus', 'police']
    dbs = {}
    for domain in domains:
        if domain in ['hospital', 'police']:
            db = '{}/{}_db.json'.format(database_dir, domain)
            with open(db, 'r', encoding='utf8') as in_f:
                dbs[domain] = json.load(in_f)
        else:
            db = '{}/{}-dbase.db'.format(database_dir, domain)
            conn = sqlite3.connect(db)
            c = conn.cursor()
            dbs[domain] = c

            # Inspect database
            print('-' * 10)
            print('Database for the {} domain:'.format(domain))
            dbs[domain].execute('SELECT name FROM sqlite_master WHERE type=\'table\';')
            print('Tables: ', dbs[domain].fetchall())
            dbs[domain].execute('SELECT * FROM {}'.format(domain))
            print(dbs[domain].description)
            rows = dbs[domain].fetchall()
            for row in rows[:10]:
                print(row)
    return dbs


def load_mwoz_databases_json(database_dir):
    """ Helper function for loading the MultiWOZ databases from json files """
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    dbs = {}
    for domain in domains:
        db = '{}/{}_db.json'.format(database_dir, domain)
        with open(db, 'r', encoding='utf8') as in_f:
            dbs[domain] = json.load(in_f)
    return dbs