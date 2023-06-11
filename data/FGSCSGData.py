from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import spacy
import os
import string
import functools
from copy import deepcopy
from utils import print_time_info


def FGSCSGData(
        data_dir, is_spacy, is_lemma, fold_attr,
        use_punct, min_length=-1, train=True):
    raw_data, sf_data = parse_data(data_dir, fold_attr, train)
    data = parse_data(raw_data, is_spacy)
    input_data, input_attr_seqs, output_labels = \
        build_dataset(data, is_lemma, use_punct, min_length)

    temp = input_data[0]
    refs_list = []
    temp_refs = []
    for i, ref in zip(input_data, output_labels):
        if i != temp:
            for _ in range(len(temp_refs)):
                refs_list.append(temp_refs)
            temp_refs = [ref]
        else:
            temp_refs.append(ref)
        temp = i
    for _ in range(len(temp_refs)):
        refs_list.append(temp_refs)

    return input_data, input_attr_seqs, output_labels, refs_list, sf_data


def parse_data(data_dir, fold_attr, train):
    if train:
        lines_file = os.path.join(data_dir, "train.csv")
    else:
        lines_file = os.path.join(data_dir, "test.csv")
    data = list()
    with open(lines_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if len(line.split('"')) >= 3 and i > 0:
                attributes = line.split('"')[1].strip('"')
                s = line.replace("\"{}\",".format(attributes), "") \
                    .replace("\n", "")
                attributes = attributes.split(',')
                attributes = [
                        [
                            i.strip().split('[')[0],
                            i.strip().split('[')[1].strip(']')
                        ] for i in attributes]
                # trim all punctuation marks in one line
                seq = functools.reduce(
                        lambda s, c: s.replace(c, ''), string.punctuation, s)
                data.append([attributes, seq])

    sf_data = []
    if fold_attr:
        for idx, d in enumerate(data):
            for a_idx, attr_pair in enumerate(d[0]):
                data[idx][0][a_idx][1] = attr_pair[1].lower()
            data[idx][1] = data[idx][1].lower()
            sf_data.append(d[0])
        sf_data_list = deepcopy(sf_data)
    for idx, d in enumerate(data):
        split_sent = d[1].split()
        data[idx][1] = ' '.join(split_sent)

    sf_data = []
    for sf in sf_data_list:
        temp = dict()
        for attr_pair in sf:
            temp[attr_pair[0]] = attr_pair[1]
        sf_data.append(temp)

    return data, sf_data


def parse_data(raw_data, is_spacy):
    data = []
    '''
    if is_spacy:
        spacy_parser = spacy.load('en')
    '''
    for idx, dialog in enumerate(raw_data):
        if idx % 1000 == 0:
            print_time_info(
                    "Processed {}/{} data".format(
                        idx, len(raw_data)))
        spacy_parsed_dialog = []
        nltk_parsed_dialog = []
        # encoder input
        spacy_parsed_dialog.append(dialog[0])
        # output label
        line = dialog[1]
        spacy_line, nltk_line = [], []
        if is_spacy:
            '''
            parsed_line = spacy_parser(line)
            spacy_line = [
                    d for d in [
                        [word.text, word.pos_]
                        for word in parsed_line] if d[0] != ' ']
            spacy_parsed_dialog.append(spacy_line)
            '''
            line = [
                [word] for word in line.split()
            ]
            spacy_parsed_dialog.append(line)
        else:
            nltk_line = pos_tag(word_tokenize(line), tagset='universal')
            nltk_line = [
                    [d[0], d[1]]
                    if d[1] != '.' else [d[0], 'PUNCT']
                    for d in nltk_line]
            nltk_parsed_dialog.append(nltk_line)

        if spacy_parsed_dialog != []:
            data.append(spacy_parsed_dialog)
        else:
            data.append(nltk_parsed_dialog)

    return data


def build_dataset(data, is_lemma, use_punct, min_length):
    input_data = []
    input_attr_seqs = []
    output_labels = []
    spacy_parser = spacy.load('en')
    for idx, dialog in enumerate(data):
        if idx % 1000 == 0:
            print_time_info("Parsed {}/{} data".format(idx, len(data)))
        attrs = []
        attrs_seq = []
        for attr_pair in dialog[0]:
            attrs_seq.append(attr_pair[0])
            attrs_seq.append(attr_pair[1])
            attrs.append('{}:{}'.format(attr_pair[0], attr_pair[1]))
        input_data.append(attrs)
        input_attr_seqs.append(attrs_seq)
        output_label = []
        for w in dialog[1]:
            output_label.append(w[0])

        output_labels.append(deepcopy(output_label))

    if min_length == -1:
        print_time_info(
                "No minimal length, data count: {}".format(len(data)))
    else:
        print_time_info("Minimal length is {}".format(min_length))
        idxs = []
        for idx, sent in enumerate(input_data):
            if len(output_labels[idx]) > min_length:
                idxs.append(idx)
        input_data = [input_data[i] for i in idxs]
        input_attr_seqs = [input_attr_seqs[i] for i in idxs]
        output_labels = [output_labels[i] for i in idxs]
        print_time_info("Data count: {}".format(len(idxs)))
    return input_data, input_attr_seqs, output_labels