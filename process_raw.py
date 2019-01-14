# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: process_raw.py

@time: 2019/1/5 17:05

@desc: process raw data

"""

import os
import codecs
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


def process_xml(file_path, is_train_file, save_folder):
    content_term, aspect_term, sentiment_term, start, end = list(), list(), list(), list(), list() # data for aspect term
    content_cate, aspect_cate, sentiment_cate = list(), list(), list()      # data for aspect category
    polarity = {'negative': 0, 'neutral': 1, 'positive': 2}

    tree = ET.parse(file_path)
    root = tree.getroot()
    for sentence in root:
        text = sentence.find('text').text.lower()
        for asp_terms in sentence.iter('aspectTerms'):
            for asp_term in asp_terms.iter('aspectTerm'):
                if asp_term.get('polarity') in polarity:
                    _text = text
                    _start = int(asp_term.get('from'))
                    _end = int(asp_term.get('to'))
                    _aspect = asp_term.get('term').lower()
                    _sentiment = polarity[asp_term.get('polarity')]
                    if _start > 0 and text[_start - 1] != ' ':
                        _text = text[:_start] + ' ' + text[_start:]
                        _start += 1
                        _end += 1
                    if _end < len(_text) and _text[_end] != ' ':
                        _text = _text[:_end] + ' ' + _text[_end:]
                    if _text[_start:_end] != _aspect:
                        raise Exception('{}=={}=={}'.format(_text, _text[_start:_end], _aspect))
                    content_term.append(_text)
                    aspect_term.append(_aspect)
                    sentiment_term.append(_sentiment)
                    start.append(_start)
                    end.append(_end)
        for asp_cates in sentence.iter('aspectCategories'):
            for asp_cate in asp_cates.iter('aspectCategory'):
                if asp_cate.get('polarity') in polarity:
                    content_cate.append(text)
                    aspect_cate.append(asp_cate.get('category'))
                    sentiment_cate.append(polarity[asp_cate.get('polarity')])

    if not os.path.exists(os.path.join(save_folder, 'term')):
        os.makedirs(os.path.join(save_folder, 'term'))

    if not is_train_file:
        test_data = {'content': content_term, 'aspect': aspect_term, 'sentiment': sentiment_term,
                     'from': start, 'to': end}
        test_data = pd.DataFrame(test_data, columns=test_data.keys())
        test_data.to_csv(os.path.join(save_folder, 'term/test.csv'), index=None)
    else:
        train_content, valid_content, train_aspect, valid_aspect, train_senti, valid_senti, train_start, valid_start, \
            train_end, valid_end = train_test_split(content_term, aspect_term, sentiment_term, start, end, test_size=0.1)
        train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti,
                      'from': train_start, 'to': train_end}
        train_data = pd.DataFrame(train_data, columns=train_data.keys())
        train_data.to_csv(os.path.join(save_folder, 'term/train.csv'), index=None)
        valid_data = {'content': valid_content, 'aspect': valid_aspect, 'sentiment': valid_senti,
                      'from': valid_start, 'to': valid_end}
        valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())
        valid_data.to_csv(os.path.join(save_folder, 'term/valid.csv'), index=None)

    if len(content_cate) > 0:
        if not os.path.exists(os.path.join(save_folder, 'category')):
            os.makedirs(os.path.join(save_folder, 'category'))

        if not is_train_file:
            test_data = {'content': content_cate, 'aspect': aspect_cate, 'sentiment': sentiment_cate}
            test_data = pd.DataFrame(test_data, columns=test_data.keys())
            test_data.to_csv(os.path.join(save_folder, 'category/test.csv'), index=None)
        else:
            train_content, valid_content, train_aspect, valid_aspect, \
                train_senti, valid_senti = train_test_split(content_cate, aspect_cate, sentiment_cate, test_size=0.1)
            train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti}
            train_data = pd.DataFrame(train_data, columns=train_data.keys())
            train_data.to_csv(os.path.join(save_folder, 'category/train.csv'), index=None)
            valid_data = {'content': valid_content, 'aspect': valid_aspect, 'sentiment': valid_senti}
            valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())
            valid_data.to_csv(os.path.join(save_folder, 'category/valid.csv'), index=None)


def process_twitter(file_path, is_train_file, save_folder):
    polarity = {'-1': 0, '0': 1, '1': 2}
    content, aspect, sentiment, start, end = list(), list(), list(), list(), list()
    with codecs.open(file_path, 'r', encoding='utf8')as reader:
        lines = reader.readlines()
        for i in range(0, len(lines), 3):
            _content = lines[i].strip().lower()
            _aspect = lines[i+1].strip().lower()
            _sentiment = lines[i+2].strip().lower()
            _start = _content.find('$t$')
            _end = _start + len(_aspect)
            content.append(_content.replace('$t$', _aspect))
            aspect.append(_aspect)
            sentiment.append(polarity[_sentiment])
            start.append(_start)
            end.append(_end)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not is_train_file:
        test_data = {'content': content, 'aspect': aspect, 'sentiment': sentiment, 'from': start, 'to': end}
        test_data = pd.DataFrame(test_data, columns=test_data.keys())
        test_data.to_csv(os.path.join(save_folder, 'test.csv'), index=None)
    else:
        train_content, valid_content, train_aspect, valid_aspect, train_senti, valid_senti, train_start, valid_start, \
            train_end, valid_end = train_test_split(content, aspect, sentiment, start, end, test_size=0.1)
        train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti,
                      'from': train_start, 'to': train_end}
        train_data = pd.DataFrame(train_data, columns=train_data.keys())
        train_data.to_csv(os.path.join(save_folder, 'train.csv'), index=None)
        valid_data = {'content': valid_content, 'aspect': valid_aspect, 'sentiment': valid_senti,
                      'from': valid_start, 'to': valid_end}
        valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())
        valid_data.to_csv(os.path.join(save_folder, 'valid.csv'), index=None)


def process_fsauor(file_path, save_path):
    folder = os.path.dirname(save_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    aspect_names = ['location_traffic_convenience', 'location_distance_from_business_district',
                    'location_easy_to_find', 'service_wait_time', 'service_waiters_attitude',
                    'service_parking_convenience', 'service_serving_speed', 'price_level',
                    'price_cost_effective', 'price_discount', 'environment_decoration', 'environment_noise',
                    'environment_space', 'environment_cleaness', 'dish_portion', 'dish_taste', 'dish_look',
                    'dish_recommendation', 'others_overall_experience', 'others_willing_to_consume_again']
    polarity = {-1: 0, 0: 1, 1: 2}
    content, aspect, sentiment = list(), list(), list()
    raw_data = pd.read_csv(file_path, header=0, index_col=0)

    for _, row in raw_data.iterrows():
        text = row['content']
        for aspect_name in aspect_names:
            if row[aspect_name] != -2:
                content.append(text)
                aspect.append(aspect_names)
                sentiment.append(polarity[row[aspect_name]])

    csv_data = {'content': content, 'aspect': aspect, 'sentiment': sentiment}
    csv_data = pd.DataFrame(csv_data, columns=csv_data.keys())
    csv_data.to_csv(save_path, index=0)


def process_bdci(file_path, is_train_file, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    raw_data = pd.read_csv(file_path, header=0, index_col=0)
    if is_train_file:
        polarity = {-1: 0, 0: 1, 1: 2}
        raw_data = raw_data[['content', 'subject', 'sentiment_value']]
        raw_data['sentiment_value'].replace(polarity, inplace=True)
        raw_data.rename(columns={'subject': 'aspect', 'sentiment_value': 'sentiment'})
        train_data, valid_data = train_test_split(raw_data, test_size=0.1)
        train_data.to_csv(os.path.join(save_folder, 'train.csv'), index=None)
        valid_data.to_csv(os.path.join(save_folder, 'valid.csv'), index=None)
    else:
        raw_data[['content']].to_csv(os.path.join(save_folder, 'test.csv'), index=None)


if __name__ == '__main__':
    process_xml('./raw_data/semeval14_laptop/Laptop_Train_v2.xml', is_train_file=True, save_folder='./data/laptop')
    process_xml('./raw_data/semeval14_laptop/Laptops_Test_Gold.xml', is_train_file=False, save_folder='./data/laptop')

    process_xml('./raw_data/semeval14_restaurant/Restaurants_Train_v2.xml', is_train_file=True,
                save_folder='./data/restaurant')
    process_xml('./raw_data/semeval14_restaurant/Restaurants_Test_Gold.xml', is_train_file=False,
                save_folder='./data/restaurant')

    process_twitter('./raw_data/twitter/train.txt', is_train_file=True, save_folder='./data/twitter')
    process_twitter('./raw_data/twitter/test.txt', is_train_file=False, save_folder='./data/twitter')

    # process_fsauor('./raw_data/fsauor2018/train.csv', save_path='./data/fsauor/train.csv')
    # process_fsauor('./raw_data/fsauor2018/valid.csv', save_path='./data/fsauor/valid.csv')
    # process_fsauor('./raw_data/fsauor2018/test.csv', save_path='./data/fsauor/test.csv')
    #
    # process_bdci('./raw_data/bdci18_car/train.csv', is_train_file=True, save_folder='./data/bdci')
    # process_bdci('./raw_data/bdci18_car/test.csv', is_train_file=True, save_folder='./data/bdci')

