## Aspect-based Sentiment Analysis

Keras implementation (tensorflow backend) of aspect based sentiment analysis

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
```
sh preprocess.sh
```

### Training
```
python3 train.py
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

### Performance
Note: results in the parenthese is the performance of models with word embeddings fixed (aspect embeddings fine tuned)

- Accuracy

| model   | laptop(paper) |  laptop         | restaurant(paper) | restaurant     | twitter(paper) | twitter          |
|---------|---------------|-----------------|-------------------|----------------|----------------|------------------|
|td_lstm  |               |  0.69122(0.7225)|                   |  0.7732(0.7875)|    0.708       |  0.69508(0.7182) |
|tc_lstm  |               |  0.68652(0.6833)|                   |  0.7642(0.7687)|    0.715       |  0.70379(0.72398)|
|ae_lstm  |    0.689      |  0.67398(0.6834)|      0.766        |  0.7598(0.7571)|                |  0.6878(0.6936)  |
|at_lstm  |               |  0.68181(0.7179)|                   |  0.7669(0.7696)|                |  0.6575(0.70520) |
|atae_lstm|    0.687      |  0.68025(0.6849)|      0.772        |  0.7598(0.7607)|                |  0.68061(0.69508)|
|memnet   |    0.7237     |  0.5329(0.52978)|      0.8095       |  0.6508(0.6508)|                |  0.57803(0.5606) |
|ram      |    0.7449     |  0.7021(0.7210) |      0.8023       |  0.7866(0.7946)|    0.6936      |  0.69653(0.71242)|
|ian      |    0.721      |  0.6912(0.6927) |      0.786        |  0.7758(0.7892)|                |  0.6835(0.71242) |
|cabasc   |    0.7507     |                 |      0.8089       |                |    0.7153      |                  |

- Macro-F1

| model   | laptop(paper) |  laptop         | restaurant(paper) | restaurant     | twitter(paper) | twitter          |
|---------|---------------|-----------------|-------------------|----------------|----------------|------------------|
|td_lstm  |               |  0.62223(0.6667)|                   |  0.6623(0.6836)|    0.690       |  0.6783(0.70238) |
|tc_lstm  |               |  0.62287(0.6223)|                   |  0.6022(0.6651)|    0.695       |  0.6797(0.70639) |
|ae_lstm  |               |  0.60334(0.6159)|                   |  0.6365(0.6300)|                |  0.6638(0.66873) |
|at_lstm  |               |  0.61957(0.6564)|                   |  0.6630(0.6451)|                |  0.6553(0.67674) |
|atae_lstm|               |  0.6172(0.63431)|                   |  0.6096(0.6430)|                |  0.6629(0.67799) |
|memnet   |               |  0.40214(0.3538)|                   |  0.3339(0.3011)|                |  0.5096(0.49457) |
|ram      |   0.7135      |  0.6474(0.6794) | 0.7080            |  0.6855(0.6915)|    0.6730      |  0.6769(0.6873)  |
|ian      |               |  0.62409(0.6306)|                   |  0.6675(0.6800)|                |  0.65373(0.70094)|
|cabasc   |               |                 |                   |                |                |                  |

- Personal conclusion
1. I found `AT-LSTM` is always better than `AE-LSTM` & `ATAE-LSTM`. Actually it's not just on SemEval14 & twitter data, but many other sentiment analysis data.
2. Surprisingly, I failed to achieved similar performance as stated in the parper of `Memnet`. Or maybe there are bugs in the code?
3. `TD-LSTM` performs unexpectedly well.
4. Models with fixed word embeddings are generally better than those with fine-tuned word embeddings, which is consistent with the paper of `RAM`.