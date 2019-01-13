#!/usr/bin/env python

'''
**Baseline methods for the 4th task of SemEval 2014**

Run a task from the terminal::
>>> python baselines.py -t file -m taskNum

or, import within python. E.g., for Aspect Term Extraction::
from baselines import *
corpus = Corpus(ET.parse(trainfile).getroot().findall('sentence'))
unseen = Corpus(ET.parse(testfile).getroot().findall('sentence'))
b1 = BaselineAspectExtractor(corpus)
predicted = b1.tag(unseen.corpus)
corpus.write_out('%s--test.predicted-aspect.xml'%domain_name, predicted, short=False)

Similarly, for Aspect Category Detection, Aspect Term Polarity Estimation, and Aspect Category Polarity Estimation.
'''

__author__ = "J. Pavlopoulos"
__credits__ = "J. Pavlopoulos, D. Galanis, I. Androutsopoulos"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "John Pavlopoulos"
__email__ = "annis@aueb.gr"

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stopwords, imported from NLTK (v 2.0.4)
stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])


def fd(counts):
    '''Given a list of occurrences (e.g., [1,1,1,2]), return a dictionary of frequencies (e.g., {1:3, 2:1}.)'''
    d = {}
    for i in counts: d[i] = d[i] + 1 if i in d else 1
    return d


freq_rank = lambda d: sorted(d, key=d.get, reverse=True)
'''Given a map, return ranked the keys based on their values.'''


def fd2(counts):
    '''Given a list of 2-uplets (e.g., [(a,pos), (a,pos), (a,neg), ...]), form a dict of frequencies of specific items (e.g., {a:{pos:2, neg:1}, ...}).'''
    d = {}
    for i in counts:
        # If the first element of the 2-uplet is not in the map, add it.
        if i[0] in d:
            if i[1] in d[i[0]]:
                d[i[0]][i[1]] += 1
            else:
                d[i[0]][i[1]] = 1
        else:
            d[i[0]] = {i[1]: 1}
    return d


def validate(filename):
    '''Validate an XML file, w.r.t. the format given in the 4th task of **SemEval '14**.'''
    elements = ET.parse(filename).getroot().findall('sentence')
    aspects = []
    for e in elements:
        for eterms in e.findall('aspectTerms'):
            if eterms is not None:
                for a in eterms.findall('aspectTerm'):
                    aspects.append(Aspect('', '', []).create(a).term)
    return elements, aspects


fix = lambda text: escape(text.encode('utf8')).replace('\"', '&quot;')
'''Simple fix for writing out text.'''

# Dice coefficient
def dice(t1, t2, stopwords=[]):
    tokenize = lambda t: set([w for w in t.split() if (w not in stopwords)])
    t1, t2 = tokenize(t1), tokenize(t2)
    return 2. * len(t1.intersection(t2)) / (len(t1) + len(t2))


class Category:
    '''Category objects contain the term and polarity (i.e., pos, neg, neu, conflict) of the category (e.g., food, price, etc.) of a sentence.'''

    def __init__(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity

    def create(self, element):
        self.term = element.attrib['category']
        self.polarity = element.attrib['polarity']
        return self

    def update(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity


class Aspect:
    '''Aspect objects contain the term (e.g., battery life) and polarity (i.e., pos, neg, neu, conflict) of an aspect.'''

    def __init__(self, term, polarity, offsets):
        self.term = term
        self.polarity = polarity
        self.offsets = offsets

    def create(self, element):
        self.term = element.attrib['term']
        self.polarity = element.attrib['polarity']
        self.offsets = {'from': str(element.attrib['from']), 'to': str(element.attrib['to'])}
        return self

    def update(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity


class Instance:
    '''An instance is a sentence, modeled out of XML (pre-specified format, based on the 4th task of SemEval 2014).
    It contains the text, the aspect terms, and any aspect categories.'''

    def __init__(self, element):
        self.text = element.find('text').text
        self.id = element.get('id')
        self.aspect_terms = [Aspect('', '', offsets={'from': '', 'to': ''}).create(e) for es in
                             element.findall('aspectTerms') for e in es if
                             es is not None]
        self.aspect_categories = [Category(term='', polarity='').create(e) for es in element.findall('aspectCategories')
                                  for e in es if
                                  es is not None]

    def get_aspect_terms(self):
        return [a.term.lower() for a in self.aspect_terms]

    def get_aspect_categories(self):
        return [c.term.lower() for c in self.aspect_categories]

    def add_aspect_term(self, term, polarity='', offsets={'from': '', 'to': ''}):
        a = Aspect(term, polarity, offsets)
        self.aspect_terms.append(a)

    def add_aspect_category(self, term, polarity=''):
        c = Category(term, polarity)
        self.aspect_categories.append(c)


class Corpus:
    '''A corpus contains instances, and is useful for training algorithms or splitting to train/test files.'''

    def __init__(self, elements):
        self.corpus = [Instance(e) for e in elements]
        self.size = len(self.corpus)
        self.aspect_terms_fd = fd([a for i in self.corpus for a in i.get_aspect_terms()])
        self.top_aspect_terms = freq_rank(self.aspect_terms_fd)
        self.texts = [t.text for t in self.corpus]

    def echo(self):
        print '%d instances\n%d distinct aspect terms' % (len(self.corpus), len(self.top_aspect_terms))
        print 'Top aspect terms: %s' % (', '.join(self.top_aspect_terms[:10]))

    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []

    def split(self, threshold=0.8, shuffle=False):
        '''Split to train/test, based on a threshold. Turn on shuffling for randomizing the elements beforehand.'''
        clone = copy.deepcopy(self.corpus)
        if shuffle: random.shuffle(clone)
        train = clone[:int(threshold * self.size)]
        test = clone[int(threshold * self.size):]
        return train, test

    def write_out(self, filename, instances, short=True):
        with open(filename, 'w') as o:
            o.write('<sentences>\n')
            for i in instances:
                o.write('\t<sentence id="%s">\n' % (i.id))
                o.write('\t\t<text>%s</text>\n' % fix(i.text))
                o.write('\t\t<aspectTerms>\n')
                if not short:
                    for a in i.aspect_terms:
                        o.write('\t\t\t<aspectTerm term="%s" polarity="%s" from="%s" to="%s"/>\n' % (
                            fix(a.term), a.polarity, a.offsets['from'], a.offsets['to']))
                o.write('\t\t</aspectTerms>\n')
                o.write('\t\t<aspectCategories>\n')
                if not short:
                    for c in i.aspect_categories:
                        o.write('\t\t\t<aspectCategory category="%s" polarity="%s"/>\n' % (fix(c.term), c.polarity))
                o.write('\t\t</aspectCategories>\n')
                o.write('\t</sentence>\n')
            o.write('</sentences>')


class BaselineAspectExtractor():
    '''Extract the aspects from a text.
    Use the aspect terms from the train data, to tag any new (i.e., unseen) instances.'''

    def __init__(self, corpus):
        self.candidates = [a.lower() for a in corpus.top_aspect_terms]

    def find_offsets_quickly(self, term, text):
        start = 0
        while True:
            start = text.find(term, start)
            if start == -1: return
            yield start
            start += len(term)

    def find_offsets(self, term, text):
        offsets = [(i, i + len(term)) for i in list(self.find_offsets_quickly(term, text))]
        return offsets

    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            i_.aspect_terms = []
            for c in set(self.candidates):
                if c in i_.text:
                    offsets = self.find_offsets(' ' + c + ' ', i.text)
                    for start, end in offsets: i_.add_aspect_term(term=c,
                                                                  offsets={'from': str(start + 1), 'to': str(end - 1)})
            clones.append(i_)
        return clones


class BaselineCategoryDetector():
    '''Detect the category (or categories) of an instance.
    For any new (i.e., unseen) instance, fetch the k-closest instances from the train data, and vote for the number of categories and the categories themselves.'''

    def __init__(self, corpus):
        self.corpus = corpus

    # Fetch k-neighbors (i.e., similar texts), using the Dice coefficient, and vote for #categories and category values
    def fetch_k_nn(self, text, k=5, multi=False):
        neighbors = dict([(i, dice(text, n, stopwords)) for i, n in enumerate(self.corpus.texts)])
        ranked = freq_rank(neighbors)
        topk = [self.corpus.corpus[i] for i in ranked[:k]]
        num_of_cats = 1 if not multi else int(sum([len(i.aspect_categories) for i in topk]) / float(k))
        cats = freq_rank(fd([c for i in topk for c in i.get_aspect_categories()]))
        categories = [cats[i] for i in range(num_of_cats)]
        return categories

    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            i_.aspect_categories = [Category(term=c) for c in self.fetch_k_nn(i.text)]
            clones.append(i_)
        return clones


class BaselineStageI():
    '''Stage I: Aspect Term Extraction and Aspect Category Detection.'''

    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2

    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            i_.aspect_categories, i_.aspect_terms = [], []
            for a in set(self.b1.candidates):
                offsets = self.b1.find_offsets(' ' + a + ' ', i_.text)
                for start, end in offsets:
                    i_.add_aspect_term(term=a, offsets={'from': str(start + 1), 'to': str(end - 1)})
            for c in self.b2.fetch_k_nn(i_.text):
                i_.aspect_categories.append(Category(term=c))
            clones.append(i_)
        return clones


class BaselineAspectPolarityEstimator():
    '''Estimate the polarity of an instance's aspects.
    This is a majority baseline.
    Form the <aspect,polarity> tuples from the train data, and measure frequencies.
    Then, given a new instance, vote for the polarities of the aspect terms (given).'''

    def __init__(self, corpus):
        self.corpus = corpus
        self.fd = fd2([(a.term, a.polarity) for i in self.corpus.corpus for a in i.aspect_terms])
        self.major = freq_rank(fd([a.polarity for i in self.corpus.corpus for a in i.aspect_terms]))[0]

    # Fetch k-neighbors (i.e., similar texts), using the Dice coefficient, and vote for aspect's polarity
    def k_nn(self, text, aspect, k=5):
        neighbors = dict([(i, dice(text, next.text, stopwords)) for i, next in enumerate(self.corpus.corpus) if
                          aspect in next.get_aspect_terms()])
        ranked = freq_rank(neighbors)
        topk = [self.corpus.corpus[i] for i in ranked[:k]]
        return freq_rank(fd([a.polarity for i in topk for a in i.aspect_terms]))

    def majority(self, text, aspect):
        if aspect not in self.fd:
            return self.major
        else:
            polarities = self.k_nn(text, aspect, k=5)
            if polarities:
                return polarities[0]
            else:
                return self.major

    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            for j in i_.aspect_terms: j.polarity = self.majority(i_.text, j.term)
            clones.append(i_)
        return clones


class BaselineAspectCategoryPolarityEstimator():
    '''Estimate the polarity of an instance's category (or categories).
    This is a majority baseline.
    Form the <category,polarity> tuples from the train data, and measure frequencies.
    Then, given a new instance, vote for the polarities of the categories (given).'''

    def __init__(self, corpus):
        self.corpus = corpus
        self.fd = fd2([(c.term, c.polarity) for i in self.corpus.corpus for c in i.aspect_categories])

    # Fetch k-neighbors (i.e., similar texts), using the Dice coefficient, and vote for aspect's polarity
    def k_nn(self, text, k=5):
        neighbors = dict([(i, dice(text, next.text, stopwords)) for i, next in enumerate(self.corpus.corpus)])
        ranked = freq_rank(neighbors)
        topk = [self.corpus.corpus[i] for i in ranked[:k]]
        return freq_rank(fd([c.polarity for i in topk for c in i.aspect_categories]))

    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            polarity = self.k_nn(i_.text)[0]
            for j in i_.aspect_categories:
                j.polarity = polarity
            clones.append(i_)
        return clones


class BaselineStageII():
    '''Stage II: Aspect Term and Aspect Category Polarity Estimation.
    Terms and categories are assumed given.'''

    # Baselines 3 and 4 are assumed given.
    def __init__(self, b3, b4):
        self.b3 = b3
        self.b4 = b4

    # Tag sentences with aspects and categories with their polarities
    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            for j in i_.aspect_terms: j.polarity = self.b3.majority(i_.text, j.term)
            for j in i_.aspect_categories: j.polarity = self.b4.majority(i_.text)
            clones.append(i_)
        return clones


class Evaluate():
    '''Evaluation methods, per subtask of the 4th task of SemEval '14.'''

    def __init__(self, correct, predicted):
        self.size = len(correct)
        self.correct = correct
        self.predicted = predicted

    # Aspect Extraction (no offsets considered)
    def aspect_extraction(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = [a.offsets for a in self.correct[i].aspect_terms]
            pre = [a.offsets for a in self.predicted[i].aspect_terms]
            common += len([a for a in pre if a in cor])
            retrieved += len(pre)
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        return p, r, f1, common, retrieved, relevant

    # Aspect Category Detection
    def category_detection(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = self.correct[i].get_aspect_categories()
            # Use set to avoid duplicates (i.e., two times the same category)
            pre = set(self.predicted[i].get_aspect_categories())
            common += len([c for c in pre if c in cor])
            retrieved += len(pre)
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + b ** 2) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        return p, r, f1, common, retrieved, relevant

    def aspect_polarity_estimation(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = [a.polarity for a in self.correct[i].aspect_terms]
            pre = [a.polarity for a in self.predicted[i].aspect_terms]
            common += sum([1 for j in range(len(pre)) if pre[j] == cor[j]])
            retrieved += len(pre)
        acc = common / retrieved
        return acc, common, retrieved

    def aspect_category_polarity_estimation(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = [a.polarity for a in self.correct[i].aspect_categories]
            pre = [a.polarity for a in self.predicted[i].aspect_categories]
            common += sum([1 for j in range(len(pre)) if pre[j] == cor[j]])
            retrieved += len(pre)
        acc = common / retrieved
        return acc, common, retrieved


def main(argv=None):
    # Parse the input
    opts, args = getopt.getopt(argv, "hg:dt:om:k:", ["help", "grammar", "train=", "task=", "test="])
    trainfile, testfile, task = None, None, 1
    use_msg = 'Use as:\n">>> python baselines.py --train file.xml --task 1|2|3|4(|5|6)"\n\nThis will parse a train set, examine whether is valid, split to train and test (80/20 %), write the new train, test and unseen test files, perform ABSA for task 1, 2, 3, or 4 (5 and 6 perform jointly tasks 1 & 2, and 3 & 4, respectively), and write out a file with the predictions.'
    if len(opts) == 0: sys.exit(use_msg)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.exit(use_msg)
        elif opt in ('-t', "--train"):
            trainfile = arg
        elif opt in ('-m', "--task"):
            task = int(arg)
        elif opt in ('-k', "--test"):
            testfile = arg

    # Examine if the file is in proper XML format for further use.
    print 'Validating the file...'
    try:
        elements, aspects = validate(trainfile)
        print 'PASSED! This corpus has: %d sentences, %d aspect term occurrences, and %d distinct aspect terms.' % (
            len(elements), len(aspects), len(list(set(aspects))))
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

    # Get the corpus and split into train/test.
    corpus = Corpus(ET.parse(trainfile).getroot().findall('sentence'))
    domain_name = 'laptops' if 'laptop' in trainfile else ('restaurants' if 'restau' in trainfile else 'absa')
    if testfile:
        traincorpus = corpus
        seen = Corpus(ET.parse(testfile).getroot().findall('sentence'))
    else:
        train, seen = corpus.split()
        # Store train/test files and clean up the test files (no aspect terms or categories are present); then, parse back the files back.
        corpus.write_out('%s--train.xml' % domain_name, train, short=False)
        traincorpus = Corpus(ET.parse('%s--train.xml' % domain_name).getroot().findall('sentence'))
        corpus.write_out('%s--test.gold.xml' % domain_name, seen, short=False)
        seen = Corpus(ET.parse('%s--test.gold.xml' % domain_name).getroot().findall('sentence'))

    corpus.write_out('%s--test.xml' % domain_name, seen.corpus)
    unseen = Corpus(ET.parse('%s--test.xml' % domain_name).getroot().findall('sentence'))

    # Perform the tasks, asked by the user and print the files with the predicted responses.
    if task == 1:
        b1 = BaselineAspectExtractor(traincorpus)
        print 'Extracting aspect terms...'
        predicted = b1.tag(unseen.corpus)
        corpus.write_out('%s--test.predicted-aspect.xml' % domain_name, predicted, short=False)
        print 'P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' % Evaluate(seen.corpus,
                                                                                                     predicted).aspect_extraction()
    if task == 2:
        print 'Detecting aspect categories...'
        b2 = BaselineCategoryDetector(traincorpus)
        predicted = b2.tag(unseen.corpus)
        print 'P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' % Evaluate(seen.corpus,
                                                                                                     predicted).category_detection()
        corpus.write_out('%s--test.predicted-category.xml' % domain_name, predicted, short=False)
    if task == 3:
        print 'Estimating aspect term polarity...'
        b3 = BaselineAspectPolarityEstimator(traincorpus)
        predicted = b3.tag(seen.corpus)
        corpus.write_out('%s--test.predicted-aspectPolar.xml' % domain_name, predicted, short=False)
        print 'Accuracy = %f, #Correct/#All: %d/%d' % Evaluate(seen.corpus, predicted).aspect_polarity_estimation()
    if task == 4:
        print 'Estimating aspect category polarity...'
        b4 = BaselineAspectCategoryPolarityEstimator(traincorpus)
        predicted = b4.tag(seen.corpus)
        print 'Accuracy = %f, #Correct/#All: %d/%d' % Evaluate(seen.corpus,
                                                               predicted).aspect_category_polarity_estimation()
        corpus.write_out('%s--test.predicted-categoryPolar.xml' % domain_name, predicted, short=False)
        # Perform tasks 1 & 2, and output an XML file with the predictions
    if task == 5:
        print 'Task 1 & 2: Aspect Term and Category Detection'
        b1 = BaselineAspectExtractor(traincorpus)
        b2 = BaselineCategoryDetector(traincorpus)
        b12 = BaselineStageI(b1, b2)
        predicted = b12.tag(unseen.corpus)
        corpus.write_out('%s--test.predicted-stageI.xml' % domain_name, predicted, short=False)
        print 'Task 1: P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' % Evaluate(
            seen.corpus, predicted).aspect_extraction()
        print 'Task 2: P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' % Evaluate(
            seen.corpus, predicted).category_detection()
        # Perform tasks 3 & 4, and output an XML file with the predictions
    if task == 6:
        print 'Aspect Term and Category Polarity Estimation'
        b3 = BaselineAspectPolarityEstimator(traincorpus)
        b4 = BaselineAspectCategoryPolarityEstimator(traincorpus)
        b34 = BaselineStageII(b3, b4)
        predicted = b34.tag(seen.corpus)
        corpus.write_out('%s--test.predicted-stageII.xml' % domain_name, predicted, short=False)
        print 'Task 3: Accuracy = %f (#Correct/#All: %d/%d)' % Evaluate(seen.corpus,
                                                                        predicted).aspect_polarity_estimation()
        print 'Task 4: Accuracy = %f (#Correct/#All: %d/%d)' % Evaluate(seen.corpus,
                                                                        predicted).aspect_category_polarity_estimation()


if __name__ == "__main__": main(sys.argv[1:])
