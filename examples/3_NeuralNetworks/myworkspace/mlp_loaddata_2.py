# -*- coding: utf-8 -*-
import numpy as np
import sys
from six.moves import cPickle
import h5py as h5
import codecs

f_data = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata/xx_sample.h5'
f_indice = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata.pkl'
f_data_head = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata/xx_head.h5'

data_file="/tmp/data/np/sample.h5"
encoding = 'utf-8'
def load_data(path=data_file, nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              end_char=1, oov_char=2, index_from=3):
    data = h5.File(path)


    return data

    # Y = [[w + index_from for w in x] + [end_char] for x in Y]
    #
    # if not nb_words:
    #     nb_words = max([max(x) for x in X])
    #
    # if not maxlen:
    #     maxlen = max(map(len, Y))
    #
    # Y = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in Y]
    #
    # X_train = X[:int(len(X) * (1 - test_split))]
    # y_train = Y[:int(len(X) * (1 - test_split))]
    #
    # X_test = X[int(len(X) * (1 - test_split)):]
    # y_test = Y[int(len(X) * (1 - test_split)):]
    #
    # return (X_train, y_train), (X_test, y_test)

def process(max_seq_lenth=20):
    args = ["/private/tmp/data/spark/data_text/part-00000"]
    filename = args[0]
    seed = 47
    f = codecs.open(filename,'r', encoding)
    ids = []
    start_of_word = 3
    y = []
    cookie_set = []
    local_set = []
    cate_set = []
    word_set = []
    word_len_set = []
    conf = {"y":y, "cookie":cookie_set, "local":local_set, "cate":cate_set, "word_len":word_len_set}
    df = h5.File("/tmp/data/np/sample.h5", "w")

    def shuffle(x):
        np.random.seed(seed)
        np.random.shuffle(x)

    def save(data, name):
        d = np.asarray(data,dtype="int32")
        shuffle(d)
        df.create_dataset(data=d, name=name)

    def collect(t5):
        click, cookie, local, cate, word = t5
        y.append(click)
        cookie_set.append(cookie)
        local_set.append(local)
        cate_set.append(cate)
        word_set.append(word)
        word_len_set.append(min(len(word), max_seq_lenth))

    def parseText(line):
        fs = line.strip().split("\t")
        if len(fs) < 5:
            return None
        else:
            click, cookie, local, cate, word = fs
            word = map(lambda x: int(x) + start_of_word, word.split(","))
            if len(word) < 1:
                return None
            else:
                return click, cookie, local, cate, word

    for line in f:
        t5 = parseText(line.strip())
        if t5 == None:
            continue
        collect(t5)

    map(lambda (k, v) : save(v, k), conf.iteritems())

    wordmx = np.zeros((len(word_set), max_seq_lenth),"int32")
    for i in range(len(word_set)):
        ll = min(max_seq_lenth, len(word_set[i]))
        wordmx[i,:ll] = word_set[i][:ll]

    shuffle(wordmx)
    df.create_dataset("word",data=wordmx)

    df.close()

def checkIndice():
    f = open(f_indice , 'rb')
    indices = cPickle.load(f)
    x,y = cPickle.load(open(f_data_head, 'rb'))
    id_local, id_cate, id_cookie, word2id = indices

    cvt = {v: k for k, v in word2id.iteritems()}
    for one in y:
        print str(map(lambda x: cvt[x], one[3]))

def sample2words(x, index_from=3, f_indice=f_indice):
    x = x - index_from
    f = open(f_indice)
    indices = cPickle.load(f)
    cvt = indices[3]
    cvt = {y:x for x,y in cvt.iteritems()}
    NOT_WORD='-'
    return [''.join(map(lambda xx: cvt[xx] if xx in cvt else NOT_WORD, ll)) for ll in x.tolist()]

def sample2topwords(x, index_from=3, f_indice=f_indice):
    x = x - index_from
    f = open(f_indice)
    indices = cPickle.load(f)
    cvt = indices[3]
    cvt = {y:x for x,y in cvt.iteritems()}
    NOT_WORD='-'
    return [''.join(map(lambda xx: cvt[xx] if xx in cvt else NOT_WORD, ll)) for ll in x.tolist()]


if __name__ == '__main__':
    process()