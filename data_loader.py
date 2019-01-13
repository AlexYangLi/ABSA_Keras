# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: data_loader.py

@time: 2019/1/5 10:03

@desc:

"""

import os
from utils import pickle_load


def load_input_data(data_folder, data_kind, level, use_text_input, use_split_text_input, use_aspect_input,
                    use_aspect_text_input, use_loc_input, use_offset_input):
    dirname = os.path.join('./data', data_folder)
    input_data = []
    if use_text_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input.pkl'.format(data_kind, level))))
    if use_split_text_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input_l.pkl'.format(data_kind, level))))
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input_r.pkl'.format(data_kind, level))))
    if use_aspect_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_aspect_input.pkl'.format(data_kind, level))))
    if use_aspect_text_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_aspect_input.pkl'.format(data_kind, level))))
    if use_loc_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_pos_input.pkl'.format(data_kind, level))))
    if use_offset_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_offset_input.pkl'.format(data_kind, level))))
    if len(input_data) == 1:
        input_data = input_data[0]
    if len(input_data) == 0:
        raise Exception('No Input!')
    return input_data


def load_label(data_folder, data_kind):
    dirname = os.path.join('./data', data_folder)
    return pickle_load(os.path.join(dirname, '{}_label.pkl'.format(data_kind)))

