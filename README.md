## Aspect-based Sentiment Analysis

Keras implementation (tensorflow backend) of aspect based sentiment analysi

### Models

- [Cabasc, WWW 2018](https://dl.acm.org/citation.cfm?id=3186001)  
Liu et al. "Content Attention Model for Aspect Based Sentiment Analysis"

- [RAM, EMNLP 2017](https://www.aclweb.org/anthology/D17-1047)  
Chen et al. "Recurrent Attention Network on Memory for Aspect Sentiment Analysis"

- [IAN, IJCAI 2017](https://arxiv.org/pdf/1709.00893.pdf)  
Ma ei al. "Interactive Attention Networks for Aspect-Level Sentiment Classification"

- [MemNet, EMNLP 2016](https://arxiv.org/pdf/1605.08900.pdf)  
Tang et al. "Aspect Level Sentiment Classification with Deep Memory Network"

- [ATAE-LSTM(AE-LSTM, AT-LSTM), EMNLP 2016](http://aclweb.org/anthology/D16-1058)  
Wang et al. "Attention-based LSTM for Aspect-level Sentiment Classification"

- [TD-LSTM(TC-LSTM), COLING 2016](https://arxiv.org/pdf/1512.01100)  
Tang et al. "Effective LSTMs for Target-Dependent Sentiment Classification"

### Preprocessing
 1. Download pre-trained word embeddings here: [glove.42B.300d.zip](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip), unzip and put it in `raw_data` directory
 2. Process raw data:
 ```
 python3 process_raw.py
 ```
 3. Prepare training, valid, testing data
 ```
 python3 preprocessing.pt
 ```

### Data Analysis

- Laptop

| item                           | size      |
|--------------------------------|-----------|
| training set                   | 2081      |
| valid set                      | 232       |
| test set                       | 638       |
| word_vocab                     | 3595      |
| char_vocab                     | 59        |
| aspect                         | 1181      |
| aspect_text_word_vocab         | 981       |
| aspect_text_char_vocab         | 45        |
| word_max_len                   | 83        |
| < word_len = 0.98              | 54        |
| word_left_max_len              | 70        |
| < word_left_len = 0.975        | 34        |
| word_right_max_len             | 78        |
| < word_right_len = 0.985       | 39        |
| char_max_len                   | 465       |
| < char_len = 0.98              | 255       |
| char_left_max_len              | 365       |
| < char_left_len = 0.98         | 177       |
| char_right_max_len             | 400       |
| < char_right_len = 0.981       | 177       |
| aspect_text_word_max_len       | 8         |
| < aspect_text_word_len = 0.99  | 6         |
| aspect_text_char_max_len       | 58        |
| < aspect_text_char_len = 0.99  | 28        |

- resturant(term)

| item                           | size      |
|--------------------------------|-----------|
| training set                   | 3241      |
| valid set                      | 361       |
| test set                       | 1120      |
| word_vocab                     | 4550      |
| char_vocab                     | 58        |
| aspect                         | 1528      |
| aspect_text_word_vocab         | 1407      |
| aspect_text_char_vocab         | 39        |
| word_max_len                   | 79        |
| < word_len = 0.973             | 43        |
| word_left_max_len              | 72        |
| < word_left_len = 0.985        | 34        |
| word_right_max_len             | 72        |
| < word_right_len = 0.985       | 34        |
| char_max_len                   | 358       |
| < char_len = 0.98              | 226       |
| char_left_max_len              | 344       |
| < char_left_len = 0.98         | 163       |
| char_right_max_len             | 326       |
| < char_right_len = 0.981       | 162       |
| aspect_text_word_max_len       | 21        |
| < aspect_text_word_len = 0.99  | 6         |
| aspect_text_char_max_len       | 115       |
| < aspect_text_char_len = 0.99  | 32        |

- resturant(category)

| item                           | size      |
|--------------------------------|-----------|
| training set                   | 3166      |
| valid set                      | 352       |
| test set                       | 973       |
| word_vocab                     | 5175      |
| char_vocab                     | 59        |
| aspect                         | 5         |
| word_max_len                   | 79        |
| < word_len = 0.971             | 35        |
| char_max_len                   | 357       |
| < char_len = 0.98              | 189       |

- twitter

| item                           | size   |
|--------------------------------|--------|
| training set                   | 5623   |
| valid set                      | 625    |
| test set                       | 692    |
| word_vocab                     | 13522  |
| char_vocab                     | 116    |
| aspect                         | 118    |
| aspect_text_word_vocab         | 176    |
| aspect_text_char_vocab         | 30     |
| word_max_len                   | 73     |
| < word_len = 0.997             | 37     |
| word_left_max_len              | 39     |
| < word_left_len = 0.985        | 28     |
| word_right_max_len             | 67     |
| < word_right_len = 0.994       | 32     |
| char_max_len                   | 188    |
| < char_len = 0.985             | 151    |
| char_left_max_len              | 156    |
| < char_left_len = 0.98         | 125    |
| char_right_max_len             | 164    |
| < char_right_len = 0.981       | 138    |
| aspect_text_word_max_len       |   3    |
| < aspect_text_word_len = 0.99  |   2    |
| aspect_text_char_max_len       |   21   |
| < aspect_text_char_len = 0.99  |   16   |

