import json
import string
import argparse

DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'bus', 'hospital', 'police', 'taxi']
DOM_TAG, SUBJ_TAG, REL_TAG, OBJ_TAG = '<|D|>', '<|S|>', '<|R|>', '<|O|>'

EXPECTED_SLOTS = {
    'restaurant': ['address', 'area', 'food', 'phone', 'postcode', 'pricerange', 'type'],
    'hotel': ['address', 'area', 'internet', 'parking', 'phone', 'postcode', 'pricerange', 'stars', 'type'],
    'attraction': ['address', 'area', 'entrance fee', 'phone', 'postcode', 'pricerange', 'type'],
    'train': ['arriveBy', 'day', 'departure', 'destination', 'duration', 'leaveAt', 'price'],
    'bus': ['arriveBy', 'day', 'departure', 'destination', 'duration', 'leaveAt', 'price']
}

REL_DICT = {
    'pricerange': 'price range',
    'arriveBy': 'arrive by',
    'leaveAt': 'leave at'
}

# NOTES:
# - 'taxi' is a very limited domain, which defines permissible phone numbers via a regular expression
# As such, only taxi types and colors are encoded as facts
# - Some facts are excluded for each domain, since they do not appear in the dialogues, such as 'location'
# - Presumably, the phrasing of templates matters, though optimizing template construction goes beyond the scope of
# this project


def _multi_word_index(full_string, sub_string):
    """ Returns the starting and final token positions of the sub_string within the string """

    words = [w.strip(string.punctuation) for w in full_string.split()]
    sub_words = [w.strip(string.punctuation) for w in sub_string.split()]

    if len(sub_words) == 1:
        final_index = len(words) - 1 - words[::-1].index(sub_words[0])
        return final_index, final_index + 1  # increment by one for use as range end point

    else:
        initial_ids = list()
        final_ids = list()
        for i, word in enumerate(words[:-len(sub_words) + 1]):
            if all(x == y for x, y in zip(sub_words, words[i:i + len(sub_words)])):
                initial_ids.append(i)
                final_ids.append(i + len(sub_words) - 1)
    return initial_ids[-1], final_ids[-1] + 1  # always return the final match, in case of multiple matches


def _prepare_restaurant_entry(db_entry, out_format):
    """ Prepares a single db entry by transforming it into the specified format """

    # Mapping from slot names to slot transcriptions within the facts
    slot_seq_dict = {
        'name': None,
        'address': 'located at',
        'area': 'area',
        'food': 'food',
        'phone': 'phone number',
        'postcode': 'postcode',
        'pricerange': 'price range',
        'type': None
    }

    # Capitalize certain slot values
    for slot in ['name', 'address', 'food']:
        if db_entry[slot] != 'unknown':
            db_entry[slot] = ' '.join([w.capitalize() for w in db_entry[slot].split()])

    output = list()
    entity_name = db_entry['name']

    atomic_templates = {
        'address': 'The restaurant {} is located at {}.',
        'area': 'The restaurant {} is located in the {} area of the city.',
        'food': 'The restaurant {} serves {} food.',
        'phone': 'The phone number of the restaurant {} is {}.',
        'postcode': 'The postcode of the restaurant {} is {}.',
        'pricerange': 'The restaurant {} is in the {} price range.',
        'type': '{} is a {}.'
    }

    composite_template = '{} is a {} that serves {} food in the {} price range. It is located at {}, in the {} area ' \
                         'of the city, in the {} postcode. Its phone number is {}.'

    if out_format in ['tuple', 'atomic']:
        for key, value in db_entry.items():
            if key in ['name', 'location', 'introduction', 'signature', 'id']:
                continue
            # Construct relational tuples
            if out_format == 'tuple':
                tuple_fact = \
                    '{} {} {} {} {} {} {} {}'.format(DOM_TAG, 'restaurant', SUBJ_TAG, entity_name, REL_TAG,
                                                     REL_DICT.get(key, key), OBJ_TAG, value)
                value_start_id, value_end_id = _multi_word_index(tuple_fact, value)
                output.append({'domain': 'restaurant',
                               'fact': tuple_fact,
                               'slot': key,
                               'slot_seq': REL_DICT.get(key, key),
                               'entity': entity_name,
                               'slot_value_ws_token_positions':
                                   [value_start_id + i for i in range(value_end_id - value_start_id)]})
            # Verbalize atomic facts
            else:
                atomic_fact = atomic_templates[key].format(entity_name, value)
                value_start_id, value_end_id = _multi_word_index(atomic_fact, value)
                output.append({'domain': 'restaurant',
                               'fact': atomic_fact,
                               'slot': key,
                               'slot_seq': slot_seq_dict[key],
                               'entity': entity_name,
                               'slot_value_ws_token_positions':
                                   [value_start_id + i for i in range(value_end_id - value_start_id)]})
    # Verbalize compound facts
    else:
        composite_fact = composite_template.format(entity_name,
                                                   db_entry['type'],
                                                   db_entry['food'],
                                                   db_entry['pricerange'],
                                                   db_entry['address'],
                                                   db_entry['area'],
                                                   db_entry['postcode'],
                                                   db_entry['phone'])
        info_positions, slot_seqs = dict(), dict()
        for key, value in db_entry.items():
            if key not in ['location', 'introduction', 'signature', 'id']:
                value_start_id, value_end_id = _multi_word_index(composite_fact, value)
                info_positions[key] = [value_start_id + i for i in range(value_end_id - value_start_id)]
                slot_seqs[key] = slot_seq_dict[key]
        output.append({'domain': 'restaurant',
                       'fact': composite_fact,
                       'slot_value_ws_token_positions': info_positions,
                       'entity': entity_name,
                       'slot_seq': slot_seqs})

    return output


def _prepare_hotel_entry(db_entry, out_format):
    """ Prepares a single db entry by transforming it into the specified format """

    # Mapping from slot names to slot transcriptions within the facts
    slot_seq_dict = {
        'name': None,
        'address': 'located at',
        'area': 'area',
        'internet': 'internet',
        'parking': 'parking',
        'phone': 'phone number',
        'postcode': 'postcode',
        'pricerange': 'price range',
        'stars': 'stars',
        'type': None
    }

    # Capitalize certain slot values
    for slot in ['name', 'address']:
        if db_entry[slot] != 'unknown':
            db_entry[slot] = ' '.join([w.capitalize() for w in db_entry[slot].split()])

    output = list()
    entity_name = db_entry['name']
    db_entry['type'] = 'guest house' if db_entry['type'] == 'guesthouse' else db_entry['type']

    atomic_templates = {
        'address': 'The hotel {} is located at {}.',
        'area': 'The hotel {} is located in the {} area of the city.',
        'internet': 'The hotel {} does{}have free internet.',
        'parking': 'The hotel {} does{}have free parking.',
        'phone': 'The phone number of the hotel {} is {}.',
        'postcode': 'The postcode of the hotel {} is {}.',
        'pricerange': 'The hotel {} is in the {} price range.',
        'stars': 'The hotel {} is rated as {} stars.',
        'type': 'The hotel {} is a {}.'
    }

    composite_template = 'The hotel {} is a {} in the {} price range. It is rated {} stars. It is located at {}, ' \
                         'in the {} area of the city, in the {} postcode. Its phone number is {}. It ' \
                         'does<parking>have free parking and it does<internet>have free internet.'

    if out_format in ['tuple', 'atomic']:
        for key, value in db_entry.items():
            if key in ['name', 'id', 'location', 'price', 'takesbookings']:
                continue
            # Construct relational tuples
            if out_format == 'tuple':
                tuple_fact = \
                    '{} {} {} {} {} {} {} {}'.format(DOM_TAG, 'hotel', SUBJ_TAG, entity_name, REL_TAG,
                                                     REL_DICT.get(key, key), OBJ_TAG, value)
                value_start_id, value_end_id = _multi_word_index(tuple_fact, value)
                output.append({'domain': 'hotel',
                               'fact': tuple_fact,
                               'slot': key,
                               'slot_seq': REL_DICT.get(key, key),
                               'entity': entity_name,
                               'slot_value_ws_token_positions':
                                   [value_start_id + i for i in range(value_end_id - value_start_id)]})
            # Verbalize atomic facts
            else:
                if key not in ['parking', 'internet']:
                    if key != 'type':
                        atomic_fact = atomic_templates[key].format(entity_name, value)
                    else:
                        atomic_fact = atomic_templates[key].format(entity_name, value)
                    value_start_id, value_end_id = _multi_word_index(atomic_fact, value)
                    output.append({'domain': 'hotel',
                                   'fact': atomic_fact,
                                   'slot': key,
                                   'slot_seq': slot_seq_dict[key],
                                   'entity': entity_name,
                                   'slot_value_ws_token_positions':
                                       [value_start_id + i for i in range(value_end_id - value_start_id)]})
                else:
                    if value == 'yes':
                        atomic_fact = atomic_templates[key].format(entity_name, ' ', value)
                        output.append({'domain': 'hotel',
                                       'fact': atomic_fact,
                                       'slot': key,
                                       'slot_seq': slot_seq_dict[key],
                                       'entity': entity_name,
                                       'slot_value_ws_token_positions':
                                           [atomic_fact.split(' ').index('does') + i for i in range(2)]})
                    else:
                        atomic_fact = atomic_templates[key].format(entity_name, ' not ', value)
                        output.append({'domain': 'hotel',
                                       'fact': atomic_fact,
                                       'slot': key,
                                       'slot_seq': slot_seq_dict[key],
                                       'entity': entity_name,
                                       'slot_value_ws_token_positions':
                                           [atomic_fact.split(' ').index('does') + i for i in range(3)]})
    # Verbalize compound facts
    else:
        composite_fact = composite_template.format(entity_name,
                                                   db_entry['type'],
                                                   db_entry['pricerange'],
                                                   db_entry['stars'],
                                                   db_entry['address'],
                                                   db_entry['area'],
                                                   db_entry['postcode'],
                                                   db_entry['phone'])
        info_positions, slot_seqs = dict(), dict()
        for key, value in db_entry.items():
            if key not in ['id', 'location', 'price', 'takesbookings', 'parking', 'internet']:
                value_start_id, value_end_id = _multi_word_index(composite_fact, value)
                info_positions[key] = [value_start_id + i for i in range(value_end_id - value_start_id)]
                slot_seqs[key] = slot_seq_dict[key]
        for key in ['parking', 'internet']:
            fact_tokens = composite_fact.split(' ')
            value_position = fact_tokens.index('does<{}>have'.format(key))
            if db_entry[key] == 'yes':
                composite_fact = composite_fact.replace('<{}>'.format(key), ' ')
                info_positions[key] = [value_position, value_position + 1]  # does have
            else:
                composite_fact = composite_fact.replace('<{}>'.format(key), ' not ')
                info_positions[key] = \
                    [value_position, value_position + 1, value_position + 2]  # 'does not have'
            slot_seqs[key] = slot_seq_dict[key]
        output.append({'domain': 'hotel',
                       'fact': composite_fact,
                       'slot_value_ws_token_positions': info_positions,
                       'entity': entity_name,
                       'slot_seq': slot_seqs})
    return output


def _prepare_attraction_entry(db_entry, out_format):
    """ Prepares a single db entry by transforming it into the specified format """

    # Mapping from slot names to slot transcriptions within the facts
    slot_seq_dict = {
        'name': None,
        'address': 'located at',
        'area': 'area',
        'entrance fee': 'entrance fee',
        'phone': 'phone number',
        'postcode': 'postcode',
        'pricerange': 'price range',
        'type': None
    }

    # Capitalize certain slot values
    for slot in ['name', 'address']:
        if db_entry[slot] != 'unknown':
            db_entry[slot] = ' '.join([w.capitalize() for w in db_entry[slot].split()])

    output = list()
    entity_name = db_entry['name']

    atomic_templates = {
        'address': 'The attraction {} is located at {}.',
        'area': 'The attraction {} is located in the {} area of the city.',
        'entrance fee': 'The entrance fee for the attraction {} is {}.',
        'phone': 'The phone number of the attraction {} is {}.',
        'postcode': 'The postcode of the attraction {} is {}.',
        'pricerange': 'The attraction {} is in the {} price range.',
        'type': 'The attraction {} is {}.'
    }

    composite_template = 'The attraction {} is <type> in the {} price range. The entrance fee is {}. It is located ' \
                         'at {}, in the {} area of the city, in the {} postcode. Its phone number is {}.'

    # Split certain type values
    if db_entry['type'] == 'swimmingpool':
        db_entry['type'] = 'swimming pool'
    if db_entry['type'] == 'concerthall':
        db_entry['type'] = 'concert hall'

    if out_format in ['tuple', 'atomic']:
        for key, value in db_entry.items():
            if key in ['name', 'location', 'openhours', 'id']:
                continue

            # Construct relational tuples
            if out_format == 'tuple':
                tuple_fact = '{} {} {} {} {} {} {} {}'.format(DOM_TAG, 'attraction', SUBJ_TAG, entity_name, REL_TAG,
                                                              REL_DICT.get(key, key), OBJ_TAG, value)
                value_start_id, value_end_id = _multi_word_index(tuple_fact, value)
                output.append({'domain': 'attraction',
                               'fact': tuple_fact,
                               'slot': key,
                               'slot_seq': REL_DICT.get(key, key),
                               'entity': entity_name,
                               'slot_value_ws_token_positions':
                                   [value_start_id + i for i in range(value_end_id - value_start_id)]})
            # Verbalize atomic facts
            else:
                if key != 'type':
                    atomic_fact = atomic_templates[key].format(entity_name, value)
                    value_start_id, value_end_id = _multi_word_index(atomic_fact, value)
                    output.append({'domain': 'attraction',
                                   'fact': atomic_fact,
                                   'slot': key,
                                   'slot_seq': slot_seq_dict[key],
                                   'entity': entity_name,
                                   'slot_value_ws_token_positions':
                                       [value_start_id + i for i in range(value_end_id - value_start_id)]})
                else:
                    add_front = 0
                    add_back = 0
                    if value == 'multiple sports':
                        atomic_fact = atomic_templates[key].format(entity_name, 'a {} venue'.format(value))
                        add_back = 1
                    elif value == 'entertainment':
                        atomic_fact = atomic_templates[key].format(entity_name, 'an {} venue'.format(value))
                        add_back = 1
                    elif value == 'architecture':
                        atomic_fact = atomic_templates[key].format(entity_name, 'a piece of {}'.format(value))
                        add_front = 2
                    else:
                        atomic_fact = atomic_templates[key].format(entity_name, 'a {}'.format(value))
                    # Also mask the determiner, as it is not consistent across all slot values
                    value_start_id, value_end_id = _multi_word_index(atomic_fact, value)
                    output.append({'domain': 'attraction',
                                   'fact': atomic_fact,
                                   'slot': key,
                                   'slot_seq': slot_seq_dict[key],
                                   'entity': entity_name,
                                   'slot_value_ws_token_positions':
                                       [(value_start_id - 1 - add_front) + i for i in
                                        range((value_end_id + add_back) - (value_start_id - 1 - add_front))]})
    # Verbalize compound facts
    else:
        composite_fact = composite_template.format(entity_name,
                                                   db_entry['pricerange'],
                                                   db_entry['entrance fee'],
                                                   db_entry['address'],
                                                   db_entry['area'],
                                                   db_entry['postcode'],
                                                   db_entry['phone'])
        add_front = 0
        add_back = 0
        if db_entry['type'] == 'multiple sports':
            composite_fact = composite_fact.replace('<type>', 'a {} venue'.format(db_entry['type']))
            add_back = 1
        elif db_entry['type'] == 'entertainment':
            composite_fact = composite_fact.replace('<type>', 'an {} venue'.format(db_entry['type']))
            add_back = 1
        elif db_entry['type'] == 'architecture':
            composite_fact = composite_fact.replace('<type>', 'a piece of {}'.format(db_entry['type']))
            add_front = 2
        else:
            composite_fact = composite_fact.replace('<type>', 'a {}'.format(db_entry['type']))
        info_positions, slot_seqs = dict(), dict()
        for key, value in db_entry.items():
            if key not in ['location', 'openhours', 'id']:
                value_start_id, value_end_id = _multi_word_index(composite_fact, value)
                if key != 'type':
                    info_positions[key] = [value_start_id + i for i in range(value_end_id - value_start_id)]
                else:
                    # Mask determiner
                    info_positions[key] = [(value_start_id - 1 - add_front) + i for i in
                                           range((value_end_id + add_back) - (value_start_id - 1 - add_front))]
                slot_seqs[key] = slot_seq_dict[key]
        output.append({'domain': 'attraction',
                       'fact': composite_fact,
                       'slot_value_ws_token_positions': info_positions,
                       'entity': entity_name,
                       'slot_seq': slot_seqs})
    return output


def _prepare_train_or_bus_entry(db_entry, out_format, is_train):
    """ Prepares a single db entry by transforming it into the specified format """

    # Mapping from slot names to slot transcriptions within the facts
    slot_seq_dict = {
        'trainID': None,
        'arriveBy': 'arrives',
        'day': 'every',
        'departure': 'departs from',
        'destination': 'destination',
        'duration': 'duration',
        'leaveAt': 'leaves at',
        'price': 'ticket price'
    }

    domain = 'train' if is_train else 'bus'

    # Capitalize certain slot values
    for slot in ['departure', 'destination', 'day']:
        if db_entry[slot] != 'unknown':
            db_entry[slot] = ' '.join([w.capitalize() for w in db_entry[slot].split()])

    output = list()
    entity_name = db_entry['trainID']

    atomic_templates_train = {
        'arriveBy': 'The {} train arrives at its destination by {}.',
        'day': 'The {} train operates every {}.',
        'departure': 'The {} train departs from {}.',
        'destination': 'The destination of the {} train is {}.',
        'duration': 'The duration of the journey with the {} train is {}.',
        'leaveAt': 'The {} train leaves at {}.',
        'price': 'The ticket price for the {} train is {}.'
    }

    atomic_templates_bus = {
        'arriveBy': 'The {} bus arrives at its destination by {}.',
        'day': 'The {} bus operates every {}.',
        'departure': 'The {} bus departs from {}.',
        'destination': 'The destination of the {} bus is {}.',
        'duration': 'The duration of the journey with the {} bus is {}.',
        'leaveAt': 'The {} bus leaves at {}.',
        'price': 'The ticket price for the {} bus is {}.'
    }

    composite_template_train = 'The {} train departs from {} every {}. It leaves at {}. Its destination is {} where ' \
                               'it arrives at {}. The duration of the journey is {}. The ticket price for this train ' \
                               'is {}.'

    composite_template_bus = 'The {} bus departs from {} every {}. It leaves at {}. Its destination is {} where ' \
                             'it arrives at {}. The duration of the journey is {}. The ticket price for this bus ' \
                             'is {}.'

    atomic_templates = atomic_templates_train if is_train else atomic_templates_bus
    composite_template = composite_template_train if is_train else composite_template_bus

    if out_format in ['tuple', 'atomic']:
        for key, value in db_entry.items():
            if key == 'trainID':
                continue
            # Construct relational tuples
            if out_format == 'tuple':
                tuple_fact = '{} {} {} {} {} {} {} {}'.format(DOM_TAG, domain, SUBJ_TAG, entity_name, REL_TAG,
                                                              REL_DICT.get(key, key), OBJ_TAG, value)
                value_start_id, value_end_id = _multi_word_index(tuple_fact, value)
                output.append({'domain': domain,
                               'fact': tuple_fact,
                               'slot': key,
                               'slot_seq': REL_DICT.get(key, key),
                               'entity': entity_name,
                               'slot_value_ws_token_positions':
                                   [value_start_id + i for i in range(value_end_id - value_start_id)]})
            # Verbalize atomic facts
            else:
                atomic_fact = atomic_templates[key].format(entity_name, value)
                value_start_id, value_end_id = _multi_word_index(atomic_fact, value)
                output.append({'domain': domain,
                               'fact': atomic_fact,
                               'slot': key,
                               'slot_seq': slot_seq_dict[key],
                               'entity': entity_name,
                               'slot_value_ws_token_positions':
                                   [value_start_id + i for i in range(value_end_id - value_start_id)]})
    # Verbalize compound facts
    else:
        composite_fact = composite_template.format(entity_name,
                                                   db_entry['departure'],
                                                   db_entry['day'],
                                                   db_entry['leaveAt'],
                                                   db_entry['destination'],
                                                   db_entry['arriveBy'],
                                                   db_entry['duration'],
                                                   db_entry['price'])
        info_positions, slot_seqs = dict(), dict()
        for key, value in db_entry.items():
            value_start_id, value_end_id = _multi_word_index(composite_fact, value)
            info_positions[key] = [value_start_id + i for i in range(value_end_id - value_start_id)]
            slot_seqs[key] = slot_seq_dict[key]
        output.append({'domain': domain,
                       'fact': composite_fact,
                       'slot_value_ws_token_positions': info_positions,
                       'entity': entity_name,
                       'slot_seq': slot_seqs})
    return output


def _prepare_hospital_entry(db_entry, out_format):
    """ Prepares a single db entry by transforming it into the specified format """

    output = list()

    # The 'hospital building' entry has a different format than that of the hospital departments
    if 'address' in db_entry.keys():
        entity_name = 'hospital'

        slot_seq_dict = {
            'address': 'located at',
            'postcode': 'postcode',
            'telephone': 'phone number'
        }

        # Tuples
        if out_format == 'tuple':
            for key, value in db_entry.items():

                tuple_fact = '{} {} {} {} {} {} {} {}'.format(DOM_TAG, 'hospital', SUBJ_TAG, entity_name, REL_TAG,
                                                              REL_DICT.get(key, key), OBJ_TAG, value)
                value_start_id, value_end_id = _multi_word_index(tuple_fact, value)
                output.append({'domain': 'hospital',
                               'fact': tuple_fact,
                               'slot': key,
                               'slot_seq': REL_DICT.get(key, key),
                               'entity': entity_name,
                               'slot_value_ws_token_positions':
                                   [value_start_id + i for i in range(value_end_id - value_start_id)]})
        # Atomic facts
        elif out_format == 'atomic':
            for key, value in db_entry.items():
                if key == 'address':
                    atomic_fact = 'The {} is located at {}.'.format(entity_name, value)
                elif key == 'postcode':
                    atomic_fact = 'The postcode of the {} is {}.'.format(entity_name, value)
                else:
                    atomic_fact = 'The phone number of the {} is {}.'.format(entity_name, value)
                value_start_id, value_end_id = _multi_word_index(atomic_fact, value)
                output.append({'domain': 'hospital',
                               'fact': atomic_fact,
                               'slot': key,
                               'slot_seq': slot_seq_dict[key],
                               'entity': entity_name,
                               'slot_value_ws_token_positions':
                                   [value_start_id + i for i in range(value_end_id - value_start_id)]})
        # Composite fact
        else:
            composite_fact = 'The {} is located at {}, in the {} postcode. Its phone number is {}.'.format(
                entity_name, db_entry['address'], db_entry['postcode'], db_entry['telephone'])
            info_positions, slot_seqs = dict(), dict()
            for key, value in db_entry.items():
                value_start_id, value_end_id = _multi_word_index(composite_fact, value)
                info_positions[key] = [value_start_id + i for i in range(value_end_id - value_start_id)]
                slot_seqs[key] = slot_seq_dict[key]
            output.append({'domain': 'hospital',
                           'fact': composite_fact,
                           'slot_value_ws_token_positions': info_positions,
                           'entity': entity_name,
                           'slot_seq': slot_seqs})

    # Department entries
    else:
        entity_name = db_entry['department']

        slot_seq_dict = {
            'department': None,
            'phone': 'phone number'
        }

        # As the hospital domain only has two slots, atomic facts are identical to composite facts
        atomic_templates = {'phone': 'The phone number for the {} department of the hospital is {}.'}
        composite_template = 'The phone number for the {} department of the hospital is {}.'

        if out_format in ['tuple', 'atomic']:
            for key, value in db_entry.items():
                if key in ['id', 'department']:
                    continue

                # Construct relational tuples
                if out_format == 'tuple':
                    tuple_fact = '{} {} {} {} {} {} {} {}'.format(
                        DOM_TAG, 'hospital', SUBJ_TAG, entity_name, REL_TAG, REL_DICT.get(key, key), OBJ_TAG, value)
                    value_start_id, value_end_id = _multi_word_index(tuple_fact, value)
                    output.append({'domain': 'hospital',
                                   'fact': tuple_fact,
                                   'slot': key,
                                   'slot_seq': REL_DICT.get(key, key),
                                   'entity': entity_name,
                                   'slot_value_ws_token_positions':
                                       [value_start_id + i for i in range(value_end_id - value_start_id)]})
                # Verbalize atomic facts
                elif out_format == 'atomic':
                    atomic_fact = atomic_templates[key].format(entity_name, value)
                    value_start_id, value_end_id = _multi_word_index(atomic_fact, value)
                    output.append({'domain': 'hospital',
                                   'fact': atomic_fact,
                                   'slot': key,
                                   'slot_seq': slot_seq_dict[key],
                                   'entity': entity_name,
                                   'slot_value_ws_token_positions':
                                       [value_start_id + i for i in range(value_end_id - value_start_id)]})
        # Verbalize compound facts
        else:
            composite_fact = composite_template.format(entity_name, db_entry['phone'])
            info_positions, slot_seqs = dict(), dict()
            for key, value in db_entry.items():
                if key not in ['id']:
                    value_start_id, value_end_id = _multi_word_index(composite_fact, value)
                    info_positions[key] = [value_start_id + i for i in range(value_end_id - value_start_id)]
                    slot_seqs[key] = slot_seq_dict[key]
            output.append({'domain': 'hospital',
                           'fact': composite_fact,
                           'slot_value_ws_token_positions': info_positions,
                           'entity': entity_name,
                           'slot_seq': slot_seqs})

    return output


def _prepare_police_entry(db_entry, out_format):
    """ Prepares a single db entry by transforming it into the specified format """

    slot_seq_dict = {
        'name': None,
        'address': 'located at',
        'phone': 'phone number'
    }

    # NOTE: Police DB only has a single entry
    output = list()

    # The 'hospital building' entry has a different format than that of the hospital departments
    entity_name = db_entry['name']

    # Tuples
    if out_format == 'tuple':
        for key, value in db_entry.items():
            if key in ['name', 'id']:
                continue
            tuple_fact = '{} {} {} {} {} {} {} {}'.format(
                DOM_TAG, 'police', SUBJ_TAG, entity_name, REL_TAG, REL_DICT.get(key, key), OBJ_TAG, value)
            value_start_id, value_end_id = _multi_word_index(tuple_fact, value)
            output.append({'domain': 'police',
                           'fact': tuple_fact,
                           'slot': key,
                           'slot_seq': REL_DICT.get(key, key),
                           'entity': entity_name,
                           'slot_value_ws_token_positions':
                               [value_start_id + i for i in range(value_end_id - value_start_id)]})
    # Atomic facts
    elif out_format == 'atomic':
        for key, value in db_entry.items():
            if key in ['name', 'id']:
                continue
            if key == 'address':
                atomic_fact = 'The {} is located at {}.'.format(entity_name, value)
            else:
                atomic_fact = 'The phone number for the {} is {}.'.format(entity_name, value)
            value_start_id, value_end_id = _multi_word_index(atomic_fact, value)
            output.append({'domain': 'police',
                           'fact': atomic_fact,
                           'slot': key,
                           'slot_seq': slot_seq_dict[key],
                           'entity': entity_name,
                           'slot_value_ws_token_positions':
                               [value_start_id + i for i in range(value_end_id - value_start_id)]})
    # Composite fact
    else:
        composite_fact = 'The {} is located at {}. Its phone number is {}.'.format(
            entity_name, db_entry['address'], db_entry['phone'])
        info_positions, slot_seqs = dict(), dict()
        for key, value in db_entry.items():
            if key == 'id':
                continue
            value_start_id, value_end_id = _multi_word_index(composite_fact, value)
            info_positions[key] = [value_start_id + i for i in range(value_end_id - value_start_id)]
            slot_seqs[key] = slot_seq_dict[key]
        output.append({'domain': 'police',
                       'fact': composite_fact,
                       'slot_value_ws_token_positions': info_positions,
                       'entity': entity_name,
                       'slot_seq': slot_seqs})
    return output


# TODO: The taxi facts differ from other domains in that they do not have unique prompts
def _prepare_taxi_database(db, out_format):
    """ Transforms the taxi catalogue into the desired sample format """

    # NOTE: Taxi-db is more of a catalogue

    output = list()
    if out_format == 'tuple':
        for taxi_type in db['taxi_types']:
            tuple_fact = '{} {} {} {} {} {} {} {}'.format(
                DOM_TAG, 'taxi', SUBJ_TAG, 'taxi', REL_TAG, 'type', OBJ_TAG, taxi_type)
            value_start_id, value_end_id = _multi_word_index(tuple_fact, taxi_type)
            output.append({'domain': 'taxi',
                           'fact': tuple_fact,
                           'slot': 'type',
                           'slot_seq': 'type',
                           'entity': 'taxi',
                           'slot_value_ws_token_positions':
                               [value_start_id + i for i in range(value_end_id - value_start_id)]})
        for taxi_color in db['taxi_colors']:
            tuple_fact = '{} {} {} {} {} {} {} {}'.format(
                DOM_TAG, 'hospital', SUBJ_TAG, 'taxi', REL_TAG, 'color', OBJ_TAG, taxi_color)
            value_start_id, value_end_id = _multi_word_index(tuple_fact, taxi_color)
            output.append({'domain': 'taxi',
                           'fact': tuple_fact,
                           'slot': 'color',
                           'slot_seq': 'color',
                           'entity': 'taxi',
                           'slot_value_ws_token_positions':
                               [value_start_id + i for i in range(value_end_id - value_start_id)]})

    elif out_format == 'atomic':
        for taxi_type in db['taxi_types']:
            atomic_fact = 'A taxi can be a {} type of car.'.format(taxi_type)
            value_start_id, value_end_id = _multi_word_index(atomic_fact, taxi_type)
            output.append({'domain': 'taxi',
                           'fact': atomic_fact,
                           'slot': 'type',
                           'slot_seq': 'type',
                           'entity': 'taxi',
                           'slot_value_ws_token_positions':
                               [value_start_id + i for i in range(value_end_id - value_start_id)]})
        for taxi_color in db['taxi_colors']:
            atomic_fact = 'A taxi can be {} in color.'.format(taxi_color)
            value_start_id, value_end_id = _multi_word_index(atomic_fact, taxi_color)
            output.append({'domain': 'taxi',
                           'fact': atomic_fact,
                           'slot': 'color',
                           'slot_seq': 'color',
                           'entity': 'taxi',
                           'slot_value_ws_token_positions':
                               [value_start_id + i for i in range(value_end_id - value_start_id)]})

    else:
        for taxi_type in db['taxi_types']:
            for taxi_color in db['taxi_colors']:
                composite_fact = 'A taxi can be a {} {}.'.format(taxi_color, taxi_type)
                info_positions, slot_seqs = dict(), dict()
                for key, value in [('type', taxi_type), ('color', taxi_color)]:
                    value_start_id, value_end_id = _multi_word_index(composite_fact, value)
                    info_positions[key] = [value_start_id + i for i in range(value_end_id - value_start_id)]
                    slot_seqs[key] = None
                output.append({'domain': 'taxi',
                               'fact': composite_fact,
                               'slot_value_ws_token_positions': info_positions,
                               'entity': 'taxi',
                               'slot_seq': slot_seqs})
    return output


def prepare_db_facts(database_dir, out_dir):
    """ Converts db facts into training data for knowledge-injection into pretrained language models, according to the
    'relational tuples', 'atomic facts', and 'composite facts' formats """

    # Read-in json database files
    databases = dict()
    for domain in DOMAINS:
        db = '{}/{}_db.json'.format(database_dir, domain)
        with open(db, 'r', encoding='utf8') as in_f:
            databases[domain] = json.load(in_f)

    # Derive training samples for knowledge-injection
    relational_tuples = {domain: list() for domain in DOMAINS}
    atomic_facts = {domain: list() for domain in DOMAINS}
    composite_facts = {domain: list() for domain in DOMAINS}
    transform_functions = {'restaurant': _prepare_restaurant_entry,
                           'hotel': _prepare_hotel_entry,
                           'attraction': _prepare_attraction_entry,
                           'train': _prepare_train_or_bus_entry,
                           'bus': _prepare_train_or_bus_entry,
                           'hospital': _prepare_hospital_entry,
                           'police': _prepare_police_entry,
                           'taxi': _prepare_taxi_database}

    # Iterate over db entries
    for domain in DOMAINS:
        if domain == 'taxi':
            relational_tuples[domain] = transform_functions[domain](databases[domain][0], 'tuple')
            atomic_facts[domain] += transform_functions[domain](databases[domain][0], 'atomic')
            composite_facts[domain] += transform_functions[domain](databases[domain][0], 'composite')
        else:
            for entry in databases[domain]:
                # Replace missing values with 'unknown'
                for key in entry.keys():
                    entry[key] = 'unknown' if entry[key] == '?' else entry[key]
                # Add expected slots not present in the database
                if domain not in ['hospital', 'police', 'taxi']:
                    for slot in EXPECTED_SLOTS[domain]:
                        if slot not in entry.keys():
                            if slot in ['internet', 'parking']:
                                entry[slot] = 'no'
                            else:
                                entry[slot] = 'unknown'

                # Build knowledge injection samples
                if domain == 'train':
                    relational_tuples[domain] += transform_functions[domain](entry, 'tuple', True)
                    atomic_facts[domain] += transform_functions[domain](entry, 'atomic', True)
                    composite_facts[domain] += transform_functions[domain](entry, 'composite', True)
                elif domain == 'bus':
                    relational_tuples[domain] += transform_functions[domain](entry, 'tuple', False)
                    atomic_facts[domain] += transform_functions[domain](entry, 'atomic', False)
                    composite_facts[domain] += transform_functions[domain](entry, 'composite', False)
                else:
                    relational_tuples[domain] += transform_functions[domain](entry, 'tuple')
                    atomic_facts[domain] += transform_functions[domain](entry, 'atomic')
                    composite_facts[domain] += transform_functions[domain](entry, 'composite')

    # Write to disk
    for table, name in [(relational_tuples, 'relational_tuples'),
                        (atomic_facts, 'atomic_facts'), 
                        (composite_facts, 'composite_facts')]:
        out_path = out_dir + '/db_inf_{}.json'.format(name)     
        with open(out_path, 'w', encoding='utf8') as out_f:
            json.dump(table, out_f, indent=3, sort_keys=True, ensure_ascii=False)
        print('Saved {} to {}'.format(name, out_path))

    # Report some stats
    print('=' * 10)
    print('Relational tuples per domain:')
    for domain in relational_tuples.keys():
        print('{} : {} samples'.format(domain, len(relational_tuples[domain])))
    print('-' * 5)
    print('Atomic facts per domain:')
    for domain in atomic_facts.keys():
        print('{} : {} samples'.format(domain, len(atomic_facts[domain])))
    print('-' * 5)
    print('Composite facts per domain:')
    for domain in composite_facts.keys():
        print('{} : {} samples'.format(domain, len(composite_facts[domain])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir', type=str, required=True,
                        help='path to the database directory')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='path to the directory that will contain the constructed samples')
    args = parser.parse_args()

    prepare_db_facts(args.database_dir, args.out_dir)
