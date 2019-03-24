# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: config.py

@time: 2019/1/5 9:59

@desc:

"""


class Config(object):
    def __init__(self):
        # input configuration
        self.data_folder = 'laptop/term'
        self.data_name = 'laptop'
        self.level = 'word'     # options are 'word' & 'char'
        self.max_len = {'laptop': {'word': 83, 'char': 465}, 'restaurant': {'word': 79, 'char': 358},
                        'twitter': {'word': 73, 'char': 188}}
        self.left_max_len = {'laptop': {'word': 70, 'char': 365}, 'restaurant': {'word': 72, 'char': 344},
                             'twitter': {'word': 39, 'char': 156}}
        self.right_max_len = {'laptop': {'word': 78, 'char': 400}, 'restaurant': {'word': 72, 'char': 326},
                              'twitter': {'word': 67, 'char': 164}}
        self.asp_max_len = {'laptop': {'word': 8, 'char': 58}, 'restaurant': {'word': 21, 'char': 115},
                            'twitter': {'word': 3, 'char': 21}}
        self.word_embed_dim = 300
        self.word_embed_trainable = False
        self.word_embed_type = 'glove'    # use what kind of word embeddings, can be pre-trained on a larger corpus or just on dataset
        self.aspect_embed_dim = 300
        self.aspect_embed_trainable = False
        self.aspect_embed_type = 'glove'     # use mean of word embeddings or just randomly initialization
        self.use_text_input = None
        self.use_text_input_l = None
        self.use_text_input_r = None
        self.use_text_input_r_with_pad = None
        self.use_aspect_input = None
        self.use_aspect_text_input = None
        self.use_loc_input = None
        self.use_offset_input = None
        self.use_mask = None
        self.is_aspect_term = True

        # model structure configuration
        self.exp_name = None
        self.model_name = None
        self.lstm_units = 300
        self.dense_units = 128

        # model training configuration
        self.dropout = 0.2
        self.batch_size = 32
        self.n_epochs = 50
        self.n_classes = 3
        self.learning_rate = 0.001
        self.optimizer = "adam"

        # model saving configuration
        self.checkpoint_dir = './ckpt'
        self.checkpoint_monitor = 'val_acc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early stopping configuration
        self.early_stopping_monitor = 'val_acc'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.early_stopping_mode = 'max'

        # elmo embedding configure
        self.use_elmo = True
        self.use_elmo_alone = False
        self.elmo_hub_url = './raw_data/tfhub_elmo_2'
        self.elmo_output_mode = 'elmo'
        self.idx2token = None
        self.idx2aspect_token = None
        self.elmo_trainable = True

    def init_input(self):
        if self.model_name == 'td_lstm':
            self.use_text_input, self.use_text_input_l, self.use_text_input_r = False, True, True
            self.use_aspect_input, self.use_aspect_text_input = False, False
            self.use_loc_input, self.use_offset_input = False, False
            self.use_text_input_r_with_pad, self.use_mask = False, False
        elif self.model_name == 'tc_lstm':
            self.use_text_input, self.use_text_input_l, self.use_text_input_r = False, True, True
            self.use_aspect_input, self.use_aspect_text_input = True, False
            self.use_loc_input, self.use_offset_input = False, False
            self.use_text_input_r_with_pad, self.use_mask = False, False
        elif self.model_name in ['at_lstm', 'ae_lstm', 'atae_lstm']:
            self.use_text_input, self.use_text_input_l, self.use_text_input_r = True, False, False
            self.use_aspect_input, self.use_aspect_text_input = True, False
            self.use_loc_input, self.use_offset_input = False, False
            self.use_text_input_r_with_pad, self.use_mask = False, False
        elif self.model_name == 'memnet':
            self.use_text_input, self.use_text_input_l, self.use_text_input_r = True, False, False
            self.use_aspect_input, self.use_aspect_text_input = True, False
            self.use_loc_input, self.use_offset_input = True, False
            self.use_text_input_r_with_pad, self.use_mask = False, False
        elif self.model_name == 'ram':
            self.use_text_input, self.use_text_input_l, self.use_text_input_r = True, False, False
            self.use_aspect_input, self.use_aspect_text_input = True, False
            self.use_loc_input, self.use_offset_input = True, True
            self.use_text_input_r_with_pad, self.use_mask = False, False
        elif self.model_name == 'ian':
            self.use_text_input, self.use_text_input_l, self.use_text_input_r = True, False, False
            self.use_aspect_input, self.use_aspect_text_input = False, True
            self.use_loc_input, self.use_offset_input = False, False
            self.use_text_input_r_with_pad, self.use_mask = False, False
        elif self.model_name == 'cabasc':
            self.use_text_input, self.use_text_input_l, self.use_text_input_r = True, True, False
            self.use_aspect_input, self.use_aspect_text_input = True, False
            self.use_loc_input, self.use_offset_input = False, False
            self.use_text_input_r_with_pad, self.use_mask = True, True
        else:
            raise ValueError('model name `{}` not understood'.format(self.model_name))

        if not self.is_aspect_term:
            self.use_loc_input = False
            self.use_offset_input = False




