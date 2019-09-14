# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2019/1/5 10:02

@desc:

"""

import os
import re
import time
import numpy as np
from config import Config
from data_loader import load_input_data, load_label
from models import SentimentModel
from utils import pickle_load
from utils import get_score_senti

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    if config.use_elmo:
        config.exp_name += '_elmo_alone_{}_mode_{}_{}'.format(config.use_elmo_alone, config.elmo_output_mode,
                                                              'update' if config.elmo_trainable else 'fix')
    if len(config.callbacks_to_add) > 0:
        callback_str = '_' + '_'.join(config.callbacks_to_add)
        callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
        config.exp_name += callback_str
    print(config.exp_name)

    model = SentimentModel(config)

    test_input = load_input_data(data_folder, 'test', level, config.use_text_input, config.use_text_input_l,
                                 config.use_text_input_r, config.use_text_input_r_with_pad, config.use_aspect_input,
                                 config.use_aspect_text_input, config.use_loc_input, config.use_offset_input,
                                 config.use_mask)
    test_label = load_label(data_folder, 'test')

    if not os.path.exists(os.path.join(config.checkpoint_dir, '%s/%s.hdf5' % (data_folder, config.exp_name))):
        start_time = time.time()
        train_input = load_input_data(data_folder, 'train', level, config.use_text_input, config.use_text_input_l,
                                      config.use_text_input_r, config.use_text_input_r_with_pad,
                                      config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                      config.use_offset_input, config.use_mask)
        train_label = load_label(data_folder, 'train')
        valid_input = load_input_data(data_folder, 'valid', level, config.use_text_input, config.use_text_input_l,
                                      config.use_text_input_r, config.use_text_input_r_with_pad,
                                      config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                      config.use_offset_input, config.use_mask)
        valid_label = load_label(data_folder, 'valid')

        '''
        Note: Here I combine the training data and validation data together, use them as training input to the model, 
              while I use test data to server as validation input. The reason behind is that i want to fully explore how 
              well can the model perform on the test data (Keras's ModelCheckpoint callback can help usesave the model 
              which perform best on validation data (here the test data)).
              But generally, we won't do that, because test data will not (and should not) be accessible during training 
              process.
        '''
        train_combine_valid_input = []
        for i in range(len(train_input)):
            train_combine_valid_input.append(train_input[i] + valid_input[i])
        train_combine_valid_label = train_label + valid_label

        model.train(train_combine_valid_input, train_combine_valid_label, test_input, test_label)

        elapsed_time = time.time() - start_time
        print('training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # print('score over valid data...')
    # model.score(valid_input, valid_label)
    print('score over test data...')

    # load the best model
    print('best single model:')
    model.load_best_model()
    model.score(test_input, test_label)

    # swa model
    swa_type = None
    if 'swa' in config.callbacks_to_add:
        swa_type = 'swa'
    elif 'swa_clr' in config.callbacks_to_add:
        swa_type = 'swa_clr'
    if swa_type:
        print('swa model:')
        model.load_swa_model(swa_type=swa_type)
        model.score(test_input, test_label)

    # ensemble model
    ensemble_type = None
    if 'sse' in config.callbacks_to_add:
        ensemble_type = 'sse'
    elif 'fge' in config.callbacks_to_add:
        ensemble_type = 'fge'
    if ensemble_type:
        print('ensemble model:')
        ensemble_predict = {}
        for model_file in os.listdir(os.path.join(config.checkpoint_dir, config.data_folder)):
            if model_file.startswith(config.exp_name + '_%s' % ensemble_type):
                match = re.match(r'(%s_%s_)([\d+])(.hdf5)' % (config.exp_name, ensemble_type), model_file)
                model_id = int(match.group(2))
                model_path = os.path.join(config.checkpoint_dir, config.data_folder, model_file)
                print('Logging Info: Loading {} ensemble model checkpoint: {}'.format(ensemble_type, model_file))
                model.load_model(model_path)
                ensemble_predict[model_id] = model.predict(test_input)
        '''
            we expect the models saved towards the end of run may have better performance than models saved earlier 
            in the run, we sort the models so that the older models ('s id) are first.
        '''
        sorted_ensemble_predict = sorted(ensemble_predict.items(), key=lambda x: x[0], reverse=True)
        model_predicts = []
        for model_id, model_predict in sorted_ensemble_predict:
            single_acc, single_f1 = get_score_senti(model_predict, test_label)
            print('single_%d_acc', single_acc)
            print('single_%d_f1', single_f1)

            model_predicts.append(model_predict)
            ensemble_acc, ensemble_f1 = get_score_senti(np.mean(np.array(model_predicts), axis=0), test_label)
            print('ensemble_%d_acc', ensemble_acc)
            print('ensemble_%d_f1', ensemble_f1)


if __name__ == '__main__':
    config = Config()
    config.use_elmo = False
    config.use_elmo_alone = False
    config.elmo_trainable = False
    config.callbacks_to_add = ['earlystopping']

    train_model('laptop/term', 'laptop', 'word', 'td_lstm')
    config.word_embed_trainable = True
    config.aspect_embed_trainable = True
    train_model('laptop/term', 'laptop', 'word', 'tc_lstm')
    train_model('laptop/term', 'laptop', 'word', 'ae_lstm')
    train_model('laptop/term', 'laptop', 'word', 'at_lstm')
    train_model('laptop/term', 'laptop', 'word', 'atae_lstm')
    train_model('laptop/term', 'laptop', 'word', 'memnet')
    train_model('laptop/term', 'laptop', 'word', 'ram')
    train_model('laptop/term', 'laptop', 'word', 'ian')
    train_model('laptop/term', 'laptop', 'word', 'cabasc')

    train_model('restaurant/term', 'restaurant', 'word', 'td_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'tc_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'ae_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'at_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'atae_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'memnet')
    train_model('restaurant/term', 'restaurant', 'word', 'ram')
    train_model('restaurant/term', 'restaurant', 'word', 'ian')
    train_model('restaurant/term', 'restaurant', 'word', 'cabasc')

    train_model('twitter', 'twitter', 'word', 'td_lstm')
    train_model('twitter', 'twitter', 'word', 'tc_lstm')
    train_model('twitter', 'twitter', 'word', 'ae_lstm')
    train_model('twitter', 'twitter', 'word', 'at_lstm')
    train_model('twitter', 'twitter', 'word', 'atae_lstm')
    train_model('twitter', 'twitter', 'word', 'memnet')
    train_model('twitter', 'twitter', 'word', 'ram')
    train_model('twitter', 'twitter', 'word', 'ian')
    train_model('twitter', 'twitter', 'word', 'cabasc')

    config.word_embed_trainable = False
    config.aspect_embed_trainable = True
    train_model('laptop/term', 'laptop', 'word', 'td_lstm')
    train_model('laptop/term', 'laptop', 'word', 'tc_lstm')
    train_model('laptop/term', 'laptop', 'word', 'ae_lstm')
    train_model('laptop/term', 'laptop', 'word', 'at_lstm')
    train_model('laptop/term', 'laptop', 'word', 'atae_lstm')
    train_model('laptop/term', 'laptop', 'word', 'memnet')
    train_model('laptop/term', 'laptop', 'word', 'ram')
    train_model('laptop/term', 'laptop', 'word', 'ian')
    train_model('laptop/term', 'laptop', 'word', 'cabasc')

    train_model('restaurant/term', 'restaurant', 'word', 'td_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'tc_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'ae_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'at_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'atae_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'memnet')
    train_model('restaurant/term', 'restaurant', 'word', 'ram')
    train_model('restaurant/term', 'restaurant', 'word', 'ian')
    train_model('restaurant/term', 'restaurant', 'word', 'cabasc')

    train_model('twitter', 'twitter', 'word', 'td_lstm')
    train_model('twitter', 'twitter', 'word', 'tc_lstm')
    train_model('twitter', 'twitter', 'word', 'ae_lstm')
    train_model('twitter', 'twitter', 'word', 'at_lstm')
    train_model('twitter', 'twitter', 'word', 'atae_lstm')
    train_model('twitter', 'twitter', 'word', 'memnet')
    train_model('twitter', 'twitter', 'word', 'ram')
    train_model('twitter', 'twitter', 'word', 'ian')
    train_model('twitter', 'twitter', 'word', 'cabasc')

    config.word_embed_trainable = False
    config.aspect_embed_trainable = False
    config.n_epochs = 7
    config.callbacks_to_add = ['swa']
    train_model('laptop/term', 'laptop', 'word', 'td_lstm')
    train_model('laptop/term', 'laptop', 'word', 'tc_lstm')
    train_model('laptop/term', 'laptop', 'word', 'ae_lstm')
    train_model('laptop/term', 'laptop', 'word', 'at_lstm')
    train_model('laptop/term', 'laptop', 'word', 'atae_lstm')
    train_model('laptop/term', 'laptop', 'word', 'memnet')
    train_model('laptop/term', 'laptop', 'word', 'ram')
    train_model('laptop/term', 'laptop', 'word', 'ian')
    train_model('laptop/term', 'laptop', 'word', 'cabasc')

    train_model('restaurant/term', 'restaurant', 'word', 'td_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'tc_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'ae_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'at_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'atae_lstm')
    train_model('restaurant/term', 'restaurant', 'word', 'memnet')
    train_model('restaurant/term', 'restaurant', 'word', 'ram')
    train_model('restaurant/term', 'restaurant', 'word', 'ian')
    train_model('restaurant/term', 'restaurant', 'word', 'cabasc')

    train_model('twitter', 'twitter', 'word', 'td_lstm')
    train_model('twitter', 'twitter', 'word', 'tc_lstm')
    train_model('twitter', 'twitter', 'word', 'ae_lstm')
    train_model('twitter', 'twitter', 'word', 'at_lstm')
    train_model('twitter', 'twitter', 'word', 'atae_lstm')
    train_model('twitter', 'twitter', 'word', 'memnet')
    train_model('twitter', 'twitter', 'word', 'ram')
    train_model('twitter', 'twitter', 'word', 'ian')
    train_model('twitter', 'twitter', 'word', 'cabasc')
