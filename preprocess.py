# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: preprocessing.py

@time: 2019/1/5 16:45

@desc:

"""
import os
import nltk
import numpy as np
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from config import Config
from utils import pickle_dump


def list_flatten(l):
    result = list()
    for item in l:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result


def build_vocabulary(corpus, start_id=1):
    corpus = list_flatten(corpus)
    return dict((word, idx) for idx, word in enumerate(set(corpus), start=start_id))


def build_embedding(corpus, vocab, embedding_dim=300):
    model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 2, embedding_dim), dtype='float32')

    count = 0
    for w, i in vocab.items():
        if w not in d:
            count += 1
            emb[i, :] = np.random.uniform(-0.1, 0.1, embedding_dim)
        else:
            emb[i, :] = weights[d[w], :]
    print('embedding out of vocabulary：', count)
    return emb


def build_glove_embedding(vocab, weights, d):
    emb = np.zeros(shape=(len(vocab) + 2, weights.shape[1]), dtype='float32')

    count = 0
    for w, i in vocab.items():
        if w not in d:
            count += 1
            emb[i, :] = np.random.uniform(-0.1, 0.1, weights.shape[1])
        else:
            emb[i, :] = weights[d[w], :]
    print('glove embedding out of vocabulary：', count)
    return emb


def build_aspect_embedding(aspect_vocab, split_func, word_vocab, word_embed):
    aspect_embed = np.random.uniform(-0.1, 0.1, [len(aspect_vocab.keys()), word_embed.shape[1]])
    count = 0
    for aspect, aspect_id in aspect_vocab.items():
        word_ids = [word_vocab.get(word, 0) for word in split_func(aspect)]
        if any(word_ids):
            avg_vector = np.mean(word_embed[word_ids], axis=0)
            aspect_embed[aspect_id] = avg_vector
        else:
            count += 1
    print('aspect embedding out of vocabulary:', count)
    return aspect_embed


def build_aspect_text_embedding(aspect_text_vocab, word_vocab, word_embed):
    aspect_text_embed = np.zeros(shape=(len(aspect_text_vocab) + 2, word_embed.shape[1]), dtype='float32')
    count = 0
    for aspect, aspect_id in aspect_text_vocab.items():
        if aspect in word_vocab:
            aspect_text_embed[aspect_id] = word_embed[word_vocab[aspect]]
        else:
            count += 1
            aspect_text_embed[aspect_id] = np.random.uniform(-0.1, 0.1, word_embed.shape[1])
    print('aspect text embedding out of vocabulary:', count)
    return aspect_text_embed


def analyze_len_distribution(train_input, valid_input, test_input):
    text_len = list()
    text_len.extend([len(l) for l in train_input])
    text_len.extend([len(l) for l in valid_input])
    text_len.extend([len(l) for l in test_input])
    max_len = np.max(text_len)
    min_len = np.min(text_len)
    avg_len = np.average(text_len)
    median_len = np.median(text_len)
    print('max len:', max_len, 'min_len', min_len, 'avg len', avg_len, 'median len', median_len)
    for i in range(int(median_len), int(max_len), 5):
        less = list(filter(lambda x: x <= i, text_len))
        ratio = len(less) / len(text_len)
        print(i, ratio)
        if ratio >= 0.99:
            break


def analyze_class_distribution(labels):
    for cls, count in Counter(labels).most_common():
        print(cls, count, count / len(labels))


def get_loc_info(l, start, end):
    pos_info = []
    offset_info =[]
    for i in range(len(l)):
        if i < start:
            pos_info.append(1 - abs(i - start) / len(l))
            offset_info.append(i - start)
        elif start <= i < end:
            pos_info.append(1.)
            offset_info.append(0.)
        else:
            pos_info.append(1 - abs(i - end + 1) / len(l))
            offset_info.append(i - end +1)
    return pos_info, offset_info


def split_text_and_get_loc_info(data, word_vocab, char_vocab, word_cut_func):
    word_input_l, word_input_r, word_input_r_with_pad, word_pos_input, word_offset_input = [], [], [], [], []
    char_input_l, char_input_r, char_input_r_with_pad, char_pos_input, char_offset_input = [], [], [], [], []
    word_mask, char_mask = [], []
    for idx, row in data.iterrows():
        text, word_list, char_list, aspect = row['content'], row['word_list'], row['char_list'], row['aspect']
        start, end = row['from'], row['to']

        char_input_l.append(list(map(lambda x: char_vocab.get(x, len(char_vocab)+1), char_list[:end])))
        char_input_r.append(list(map(lambda x: char_vocab.get(x, len(char_vocab)+1), char_list[start:])))
        char_input_r_with_pad.append([char_vocab.get(char_list[i], len(char_vocab)+1) if i >= start else 0
                                      for i in range(len(char_list))])  # replace left sequence with 0
        _char_mask = [1] * len(char_list)
        _char_mask[start:end] = [0.5] * (end-start)     # 1 for context, 0.5 for aspect
        char_mask.append(_char_mask)
        _pos_input, _offset_input = get_loc_info(char_list, start, end)
        char_pos_input.append(_pos_input)
        char_offset_input.append(_offset_input)

        word_list_l = word_cut_func(text[:start])
        word_list_r = word_cut_func(text[end:])
        start = len(word_list_l)
        end = len(word_list) - len(word_list_r)
        if word_list[start:end] != word_cut_func(aspect):
            if word_list[start-1:end] == word_cut_func(aspect):
                start -= 1
            elif word_list[start:end+1] == word_cut_func(aspect):
                end += 1
            else:
                raise Exception('Can not find aspect `{}` in `{}`, word list : `{}`'.format(aspect, text, word_list))
        word_input_l.append(list(map(lambda x: word_vocab.get(x, len(word_vocab)+1), word_list[:end])))
        word_input_r.append(list(map(lambda x: word_vocab.get(x, len(word_vocab)+1), word_list[start:])))
        word_input_r_with_pad.append([word_vocab.get(word_list[i], len(word_vocab) + 1) if i >= start else 0
                                      for i in range(len(word_list))])      # replace left sequence with 0
        _word_mask = [1] * len(word_list)
        _word_mask[start:end] = [0.5] * (end - start)  # 1 for context, 0.5 for aspect
        word_mask.append(_word_mask)
        _pos_input, _offset_input = get_loc_info(word_list, start, end)
        word_pos_input.append(_pos_input)
        word_offset_input.append(_offset_input)
    return (word_input_l, word_input_r, word_input_r_with_pad, word_mask, word_pos_input, word_offset_input,
            char_input_l, char_input_r, char_input_r_with_pad, char_mask, char_pos_input, char_offset_input)


def pre_process(file_folder, word_cut_func, is_en):
    print('preprocessing: ', file_folder)
    train_data = pd.read_csv(os.path.join(file_folder, 'train.csv'), header=0, index_col=None)
    train_data['word_list'] = train_data['content'].apply(word_cut_func)
    train_data['char_list'] = train_data['content'].apply(lambda x: list(x))
    train_data['aspect_word_list'] = train_data['aspect'].apply(word_cut_func)
    train_data['aspect_char_list'] = train_data['aspect'].apply(lambda x: list(x))

    valid_data = pd.read_csv(os.path.join(file_folder, 'valid.csv'), header=0, index_col=None)
    valid_data['word_list'] = valid_data['content'].apply(word_cut_func)
    valid_data['char_list'] = valid_data['content'].apply(lambda x: list(x))
    valid_data['aspect_word_list'] = valid_data['aspect'].apply(word_cut_func)
    valid_data['aspect_char_list'] = valid_data['aspect'].apply(lambda x: list(x))

    test_data = pd.read_csv(os.path.join(file_folder, 'test.csv'), header=0, index_col=None)
    test_data['word_list'] = test_data['content'].apply(word_cut_func)
    test_data['char_list'] = test_data['content'].apply(lambda x: list(x))
    test_data['aspect_word_list'] = test_data['aspect'].apply(word_cut_func)
    test_data['aspect_char_list'] = test_data['aspect'].apply(lambda x: list(x))

    print('size of training set:', len(train_data))
    print('size of valid set:', len(valid_data))
    print('size of test set:', len(test_data))

    word_corpus = np.concatenate((train_data['word_list'].values, valid_data['word_list'].values,
                                  test_data['word_list'].values)).tolist()
    char_corpus = np.concatenate((train_data['char_list'].values, valid_data['char_list'].values,
                                  test_data['char_list'].values)).tolist()
    aspect_corpus = np.concatenate((train_data['aspect'].values, valid_data['aspect'].values,
                                    test_data['aspect'].values)).tolist()
    aspect_text_word_corpus = np.concatenate((train_data['aspect_word_list'].values,
                                              valid_data['aspect_word_list'].values,
                                              test_data['aspect_word_list'].values)).tolist()
    aspect_text_char_corpus = np.concatenate((train_data['aspect_char_list'].values,
                                              valid_data['aspect_char_list'].values,
                                              test_data['aspect_char_list'].values)).tolist()

    # build vocabulary
    print('building vocabulary...')
    word_vocab = build_vocabulary(word_corpus, start_id=1)
    char_vocab = build_vocabulary(char_corpus, start_id=1)
    aspect_vocab = build_vocabulary(aspect_corpus, start_id=0)
    aspect_text_word_vocab = build_vocabulary(aspect_text_word_corpus, start_id=1)
    aspect_text_char_vocab = build_vocabulary(aspect_text_char_corpus, start_id=1)
    pickle_dump(word_vocab, os.path.join(file_folder, 'word_vocab.pkl'))
    pickle_dump(char_vocab, os.path.join(file_folder, 'char_vocab.pkl'))
    pickle_dump(aspect_vocab, os.path.join(file_folder, 'aspect_vocab.pkl'))
    pickle_dump(aspect_text_word_vocab, os.path.join(file_folder, 'aspect_text_word_vocab.pkl'))
    pickle_dump(aspect_text_char_vocab, os.path.join(file_folder, 'aspect_text_char_vocab.pkl'))
    print('finished building vocabulary!')
    print('len of word vocabulary:', len(word_vocab))
    print('sample of word vocabulary:', list(word_vocab.items())[:10])
    print('len of char vocabulary:', len(char_vocab))
    print('sample of char vocabulary:', list(char_vocab.items())[:10])
    print('len of aspect vocabulary:', len(aspect_vocab))
    print('sample of aspect vocabulary:', list(aspect_vocab.items())[:10])
    print('len of aspect text word vocabulary:', len(aspect_text_word_vocab))
    print('sample of aspect text word vocabulary:', list(aspect_text_word_vocab.items())[:10])
    print('len of aspect text char vocabulary:', len(aspect_text_char_vocab))
    print('sample of aspect text char vocabulary:', list(aspect_text_char_vocab.items())[:10])

    # prepare embedding
    print('preparing embedding...')
    word_w2v = build_embedding(word_corpus, word_vocab, config.word_embed_dim)
    aspect_word_w2v = build_aspect_embedding(aspect_vocab, word_cut_func, word_vocab, word_w2v)
    aspect_text_word_w2v = build_aspect_text_embedding(aspect_text_word_vocab, word_vocab, word_w2v)
    char_w2v = build_embedding(char_corpus, char_vocab, config.word_embed_dim)
    aspect_char_w2v = build_aspect_embedding(aspect_vocab, lambda x: list(x), char_vocab, char_w2v)
    aspect_text_char_w2v = build_aspect_text_embedding(aspect_text_char_vocab, char_vocab, char_w2v)
    np.save(os.path.join(file_folder, 'word_w2v.npy'), word_w2v)
    np.save(os.path.join(file_folder, 'aspect_word_w2v.npy'), aspect_word_w2v)
    np.save(os.path.join(file_folder, 'aspect_text_word_w2v.npy'), aspect_text_word_w2v)
    np.save(os.path.join(file_folder, 'char_w2v.npy'), char_w2v)
    np.save(os.path.join(file_folder, 'aspect_char_w2v.npy'), aspect_char_w2v)
    np.save(os.path.join(file_folder, 'aspect_text_char_w2v.npy'), aspect_text_char_w2v)

    print('finished preparing embedding!')
    print('shape of word_w2v:', word_w2v.shape)
    print('sample of word_w2v:', word_w2v[:2, :5])
    print('shape of char_w2v:', char_w2v.shape)
    print('sample of char_w2v:', char_w2v[:2, :5])
    print('shape of aspect_word_w2v:', aspect_word_w2v.shape)
    print('sample of aspect_word_w2v:', aspect_word_w2v[:2, :5])
    print('shape of aspect_char_w2v:', aspect_char_w2v.shape)
    print('sample of aspect_char_w2v:', aspect_char_w2v[:2, :5])
    print('shape of aspect_text_word_w2v:', aspect_text_word_w2v.shape)
    print('sample of aspect_text_word_w2v:', aspect_text_word_w2v[:2, :5])
    print('shape of aspect_text_char_w2v:', aspect_text_char_w2v.shape)
    print('sample of aspect_text_char_w2v:', aspect_text_char_w2v[:2, :5])

    if is_en:
        word_glove = build_glove_embedding(word_vocab, glove_weights, glove_d)
        aspect_word_glove = build_aspect_embedding(aspect_vocab, word_cut_func, word_vocab, word_glove)
        aspect_text_word_glove = build_aspect_text_embedding(aspect_text_word_vocab, word_vocab, word_glove)
        np.save(os.path.join(file_folder, 'word_glove.npy'), word_glove)
        np.save(os.path.join(file_folder, 'aspect_word_glove.npy'), aspect_word_glove)
        np.save(os.path.join(file_folder, 'aspect_text_word_glove.npy'), aspect_text_word_glove)
        print('shape of word_glove:', word_glove.shape)
        print('sample of word_glove:', word_glove[:2, :5])
        print('shape of aspect_word_glove:', aspect_word_glove.shape)
        print('sample of aspect_word_glove:', aspect_word_glove[:2, :5])
        print('shape of aspect_text_word_glove:', aspect_text_word_glove.shape)
        print('sample of aspect_text_word_glove:', aspect_text_word_glove[:2, :5])

    # prepare input
    print('preparing text input...')
    train_word_input = train_data['word_list'].apply(
        lambda x: [word_vocab.get(word, len(word_vocab)+1) for word in x]).values.tolist()
    train_char_input = train_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
    valid_word_input = valid_data['word_list'].apply(
        lambda x: [word_vocab.get(word, len(word_vocab)+1) for word in x]).values.tolist()
    valid_char_input = valid_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
    test_word_input = test_data['word_list'].apply(
        lambda x: [word_vocab.get(word, len(word_vocab)+1) for word in x]).values.tolist()
    test_char_input = test_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
    pickle_dump(train_word_input, os.path.join(file_folder, 'train_word_input.pkl'))
    pickle_dump(train_char_input, os.path.join(file_folder, 'train_char_input.pkl'))
    pickle_dump(valid_word_input, os.path.join(file_folder, 'valid_word_input.pkl'))
    pickle_dump(valid_char_input, os.path.join(file_folder, 'valid_char_input.pkl'))
    pickle_dump(test_word_input, os.path.join(file_folder, 'test_word_input.pkl'))
    pickle_dump(test_char_input, os.path.join(file_folder, 'test_char_input.pkl'))
    print('finished preparing text input!')
    print('length analysis of text word input:')
    analyze_len_distribution(train_word_input, valid_word_input, test_word_input)
    print('length analysis of text char input')
    analyze_len_distribution(train_char_input, valid_char_input, test_char_input)

    print('preparing aspect input...')
    train_aspect_input = train_data['aspect'].apply(lambda x: [aspect_vocab[x]]).values.tolist()
    valid_aspect_input = valid_data['aspect'].apply(lambda x: [aspect_vocab[x]]).values.tolist()
    test_aspect_input = test_data['aspect'].apply(lambda x: [aspect_vocab[x]]).values.tolist()
    pickle_dump(train_aspect_input, os.path.join(file_folder, 'train_aspect_input.pkl'))
    pickle_dump(valid_aspect_input, os.path.join(file_folder, 'valid_aspect_input.pkl'))
    pickle_dump(test_aspect_input, os.path.join(file_folder, 'test_aspect_input.pkl'))
    print('finished preparing aspect input!')

    print('preparing aspect text input...')
    train_aspect_text_word_input = train_data['aspect_word_list'].apply(
        lambda x: [aspect_text_word_vocab.get(word, len(aspect_text_word_vocab) + 1) for word in x]).values.tolist()
    train_aspect_text_char_input = train_data['aspect_char_list'].apply(
        lambda x: [aspect_text_char_vocab.get(char, len(aspect_text_char_vocab) + 1) for char in x]).values.tolist()
    valid_aspect_text_word_input = valid_data['aspect_word_list'].apply(
        lambda x: [aspect_text_word_vocab.get(word, len(aspect_text_word_vocab) + 1) for word in x]).values.tolist()
    valid_aspect_text_char_input = valid_data['aspect_char_list'].apply(
        lambda x: [aspect_text_char_vocab.get(char, len(aspect_text_char_vocab) + 1) for char in x]).values.tolist()
    test_aspect_text_word_input = test_data['aspect_word_list'].apply(
        lambda x: [aspect_text_word_vocab.get(word, len(aspect_text_word_vocab) + 1) for word in x]).values.tolist()
    test_aspect_text_char_input = test_data['aspect_char_list'].apply(
        lambda x: [aspect_text_char_vocab.get(char, len(aspect_text_char_vocab) + 1) for char in x]).values.tolist()
    pickle_dump(train_aspect_text_word_input, os.path.join(file_folder, 'train_word_aspect_input.pkl'))
    pickle_dump(train_aspect_text_char_input, os.path.join(file_folder, 'train_char_aspect_input.pkl'))
    pickle_dump(valid_aspect_text_word_input, os.path.join(file_folder, 'valid_word_aspect_input.pkl'))
    pickle_dump(valid_aspect_text_char_input, os.path.join(file_folder, 'valid_char_aspect_input.pkl'))
    pickle_dump(test_aspect_text_word_input, os.path.join(file_folder, 'test_word_aspect_input.pkl'))
    pickle_dump(test_aspect_text_char_input, os.path.join(file_folder, 'test_char_aspect_input.pkl'))
    print('finished preparing aspect text input!')
    print('length analysis of aspect text word input:')
    analyze_len_distribution(train_aspect_text_word_input, valid_aspect_text_word_input, test_aspect_text_word_input)
    print('length analysis of aspect text char input')
    analyze_len_distribution(train_aspect_text_char_input, valid_aspect_text_char_input, test_aspect_text_char_input)

    if 'from' in train_data.columns:
        print('preparing left text input, right text input & position input...')
        train_word_input_l, train_word_input_r, train_word_input_r_with_pad, train_word_mask, train_word_pos_input, \
            train_word_offset_input, train_char_input_l, train_char_input_r, train_char_input_r_with_pad, \
            train_char_mask, train_char_pos_input, train_char_offset_input = split_text_and_get_loc_info(train_data,
                                                                                                         word_vocab,
                                                                                                         char_vocab,
                                                                                                         word_cut_func)
        pickle_dump(train_word_input_l, os.path.join(file_folder, 'train_word_input_l.pkl'))
        pickle_dump(train_word_input_r, os.path.join(file_folder, 'train_word_input_r.pkl'))
        pickle_dump(train_word_input_r_with_pad, os.path.join(file_folder, 'train_word_input_r_with_pad.pkl'))
        pickle_dump(train_word_mask, os.path.join(file_folder, 'train_word_mask.pkl'))
        pickle_dump(train_word_pos_input, os.path.join(file_folder, 'train_word_pos_input.pkl'))
        pickle_dump(train_word_offset_input, os.path.join(file_folder, 'train_word_offset_input.pkl'))
        pickle_dump(train_char_input_l, os.path.join(file_folder, 'train_char_input_l.pkl'))
        pickle_dump(train_char_input_r, os.path.join(file_folder, 'train_char_input_r.pkl'))
        pickle_dump(train_char_input_r_with_pad, os.path.join(file_folder, 'train_char_input_r_with_pad.pkl'))
        pickle_dump(train_char_mask, os.path.join(file_folder, 'train_char_mask.pkl'))
        pickle_dump(train_char_pos_input, os.path.join(file_folder, 'train_char_pos_input.pkl'))
        pickle_dump(train_char_offset_input, os.path.join(file_folder, 'train_char_offset_input.pkl'))

        valid_word_input_l, valid_word_input_r, valid_word_input_r_with_pad, valid_word_mask, valid_word_pos_input, \
            valid_word_offset_input, valid_char_input_l, valid_char_input_r, valid_char_input_r_with_pad, \
            valid_char_mask, valid_char_pos_input, valid_char_offset_input = split_text_and_get_loc_info(valid_data,
                                                                                                         word_vocab,
                                                                                                         char_vocab,
                                                                                                         word_cut_func)
        pickle_dump(valid_word_input_l, os.path.join(file_folder, 'valid_word_input_l.pkl'))
        pickle_dump(valid_word_input_r, os.path.join(file_folder, 'valid_word_input_r.pkl'))
        pickle_dump(valid_word_input_r_with_pad, os.path.join(file_folder, 'valid_word_input_r_with_pad.pkl'))
        pickle_dump(valid_word_mask, os.path.join(file_folder, 'valid_word_mask.pkl'))
        pickle_dump(valid_word_pos_input, os.path.join(file_folder, 'valid_word_pos_input.pkl'))
        pickle_dump(valid_word_offset_input, os.path.join(file_folder, 'valid_word_offset_input.pkl'))
        pickle_dump(valid_char_input_l, os.path.join(file_folder, 'valid_char_input_l.pkl'))
        pickle_dump(valid_char_input_r, os.path.join(file_folder, 'valid_char_input_r.pkl'))
        pickle_dump(valid_char_input_r_with_pad, os.path.join(file_folder, 'valid_char_input_r_with_pad.pkl'))
        pickle_dump(valid_char_mask, os.path.join(file_folder, 'valid_char_mask.pkl'))
        pickle_dump(valid_char_pos_input, os.path.join(file_folder, 'valid_char_pos_input.pkl'))
        pickle_dump(valid_char_offset_input, os.path.join(file_folder, 'valid_char_offset_input.pkl'))

        test_word_input_l, test_word_input_r, test_word_input_r_with_pad, test_word_mask, test_word_pos_input, \
            test_word_offset_input, test_char_input_l, test_char_input_r, test_char_input_r_with_pad, test_char_mask, \
            test_char_pos_input, test_char_offset_input = split_text_and_get_loc_info(test_data, word_vocab,
                                                                                      char_vocab, word_cut_func)
        pickle_dump(test_word_input_l, os.path.join(file_folder, 'test_word_input_l.pkl'))
        pickle_dump(test_word_input_r, os.path.join(file_folder, 'test_word_input_r.pkl'))
        pickle_dump(test_word_input_r_with_pad, os.path.join(file_folder, 'test_word_input_r_with_pad.pkl'))
        pickle_dump(test_word_mask, os.path.join(file_folder, 'test_word_mask.pkl'))
        pickle_dump(test_word_pos_input, os.path.join(file_folder, 'test_word_pos_input.pkl'))
        pickle_dump(test_word_offset_input, os.path.join(file_folder, 'test_word_offset_input.pkl'))
        pickle_dump(test_char_input_l, os.path.join(file_folder, 'test_char_input_l.pkl'))
        pickle_dump(test_char_input_r, os.path.join(file_folder, 'test_char_input_r.pkl'))
        pickle_dump(test_char_input_r_with_pad, os.path.join(file_folder, 'test_char_input_r_with_pad.pkl'))
        pickle_dump(test_char_mask, os.path.join(file_folder, 'test_char_mask.pkl'))
        pickle_dump(test_char_pos_input, os.path.join(file_folder, 'test_char_pos_input.pkl'))
        pickle_dump(test_char_offset_input, os.path.join(file_folder, 'test_char_offset_input.pkl'))

        print('length analysis of left text word input:')
        analyze_len_distribution(train_word_input_l, valid_word_input_l, test_word_input_l)
        print('length analysis of left text char input')
        analyze_len_distribution(train_char_input_l, valid_char_input_l, test_char_input_l)
        print('length analysis of right text word input:')
        analyze_len_distribution(train_word_input_r, valid_word_input_r, test_word_input_r)
        print('length analysis of right text char input')
        analyze_len_distribution(train_char_input_r, valid_char_input_r, test_char_input_r)

    # prepare output
    print('preparing output....')
    pickle_dump(train_data['sentiment'].values.tolist(), os.path.join(file_folder, 'train_label.pkl'))
    pickle_dump(valid_data['sentiment'].values.tolist(), os.path.join(file_folder, 'valid_label.pkl'))
    if 'sentiment' in test_data.columns:
        pickle_dump(test_data['sentiment'].values.tolist(), os.path.join(file_folder, 'test_label.pkl'))
    print('finished preparing output!')
    print('class analysis of training set:')
    analyze_class_distribution(train_data['sentiment'].values.tolist())
    print('class analysis of valid set:')
    analyze_class_distribution(valid_data['sentiment'].values.tolist())
    if 'sentiment' in test_data.columns:
        print('class analysis of test set:')
        analyze_class_distribution(valid_data['sentiment'].values.tolist())


if __name__ == '__main__':
    config = Config()
    glove_model = KeyedVectors.load_word2vec_format('./raw_data/glove.42B.300d.txt', binary=False)
    glove_weights = glove_model.wv.syn0
    glove_d = dict([(k, v.index) for k, v in glove_model.wv.vocab.items()])

    pre_process('./data/laptop/term', lambda x: nltk.word_tokenize(x), True)
    pre_process('./data/restaurant/term', lambda x: nltk.word_tokenize(x), True)
    pre_process('./data/restaurant/category', lambda x: nltk.word_tokenize(x), True)
    pre_process('./data/twitter', lambda x: nltk.word_tokenize(x), True)
