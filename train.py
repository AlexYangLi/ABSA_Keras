# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2019/1/5 10:02

@desc:

"""

import os
import time
from config import Config
from data_loader import load_input_data, load_label
from models import SentimentModel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train_model(data_folder, data_name, level, model_name, is_aspect_term=True):
    config.data_folder = data_folder
    config.data_name = data_name
    if not os.path.exists(os.path.join(config.checkpoint_dir, data_folder)):
        os.makedirs(os.path.join(config.checkpoint_dir, data_folder))
    config.level = level
    config.model_name = model_name
    config.is_aspect_term = is_aspect_term
    config.init_input()
    config.exp_name = '{}_{}_wv_{}'.format(model_name, level, config.word_embed_type)
    config.exp_name = config.exp_name + '_update' if config.word_embed_trainable else config.exp_name + '_fix'
    if config.use_aspect_input:
        config.exp_name += '_aspv_{}'.format(config.aspect_embed_type)
        config.exp_name = config.exp_name + '_update' if config.aspect_embed_trainable else config.exp_name + '_fix'

    print(config.exp_name)
    model = SentimentModel(config)

    valid_input = load_input_data(data_folder, 'valid', level, config.use_text_input, config.use_split_text_input,
                                  config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                  config.use_offset_input)
    valid_label = load_label(data_folder, 'valid')
    test_input = load_input_data(data_folder, 'test', level, config.use_text_input, config.use_split_text_input,
                                 config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                 config.use_offset_input)
    test_label = load_label(data_folder, 'test')

    if not os.path.exists(os.path.join(config.checkpoint_dir, '%s/%s.hdf5' % (data_folder, config.exp_name))):
        start_time = time.time()

        train_input = load_input_data(data_folder, 'train', level, config.use_text_input, config.use_split_text_input,
                                      config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                      config.use_offset_input)
        train_label = load_label(data_folder, 'train')
        model.train(train_input, train_label, valid_input, valid_label)

        elapsed_time = time.time() - start_time
        print('training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # load the best model
    model.load()

    print('score over valid data...')
    model.score(valid_input, valid_label)
    print('score over test data...')
    model.score(test_input, test_label)


if __name__ == '__main__':
    config = Config()
    # train_model('laptop/term', 'laptop', 'word', 'td_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'tc_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'ae_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'at_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'atae_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'memnet')
    # config.word_embed_trainable = False
    # train_model('laptop/term', 'laptop', 'word', 'ram')
    # config.word_embed_trainable = True
    # train_model('laptop/term', 'laptop', 'word', 'ram')
    train_model('laptop/term', 'laptop', 'word', 'ian')

    # train_model('restaurant/term', 'restaurant', 'word', 'td_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'tc_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'ae_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'at_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'atae_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'memnet')
    # config.word_embed_trainable = False
    # train_model('restaurant/term', 'restaurant', 'word', 'ram')
    # config.word_embed_trainable = True
    # train_model('restaurant/term', 'restaurant', 'word', 'ram')
    # train_model('restaurant/term', 'restaurant', 'word', 'ian')

    # train_model('twitter/term', 'twitter', 'word', 'td_lstm')
    # train_model('twitter/term', 'twitter', 'word', 'tc_lstm')
    # train_model('twitter/term', 'twitter', 'word', 'ae_lstm')
    # train_model('twitter/term', 'twitter', 'word', 'at_lstm')
    # train_model('twitter/term', 'twitter', 'word', 'atae_lstm')
    # train_model('twitter/term', 'twitter', 'word', 'memnet')
    # config.word_embed_trainable = False
    # train_model('twitter/term', 'twitter', 'word', 'ram')
    # config.word_embed_trainable = True
    # train_model('twitter/term', 'twitter', 'word', 'ram')
    # train_model('twitter/term', 'twitter', 'word', 'ian')

