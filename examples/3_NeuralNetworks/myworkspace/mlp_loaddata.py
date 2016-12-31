# -*- coding: utf-8 -*-
import numpy as np
import sys
from six.moves import cPickle
import codecs

f_data = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata/sample.pkl'
f_indice = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata/indices.pkl'
f_data_head = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata/sample.head.pkl'
encoding = 'utf-8'

def load_data(path=f_data_head, nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              end_char=1, oov_char=2, index_from=3):
    f = open(path, 'rb')
    X, Y = cPickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)

    Y = [[w + index_from for w in x] + [end_char] for x in Y]

    if not nb_words:
        nb_words = max([max(x) for x in X])

    if not maxlen:
        maxlen = max(map(len, Y))

    Y = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in Y]

    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = Y[:int(len(X) * (1 - test_split))]

    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = Y[int(len(X) * (1 - test_split)):]

    return (X_train, y_train), (X_test, y_test)

def process():
    args = ["/Users/hehehe/Downloads/sample"]
    filename = args[0]

    f = codecs.open(filename,'r', encoding)
    x = []
    y = []
    cookie2id = {}
    local2id = {}
    cate2id = {}
    word2id = {}

    def toID(xx2id, x):
        xx2id.setdefault(x, len(xx2id))
        return xx2id[x]

    def parseText(line):
        fs = line.split("\t")
        if len(fs) != 4:
            None
        else:
            loc, cate, cookie, title = fs
            loc = int(loc)
            cate = int(cate)
            if cookie[-2:-1] == "==":
                cookie = cookie[:-3]
            if loc != '-' or cate != "-" or title != "":
                return (loc, cate, cookie, title)
            else:
                return None

    for line in f:
        fs = parseText(line.strip())
        if fs == None:
            continue
        loc, cate, cookie, title = fs

        id_local = toID(local2id, loc)
        id_cate = toID(cate2id, cate)
        id_cookie = toID(cookie2id, cookie)
        t1 = map(lambda x: toID(word2id,x), title)

        x.append([id_local, id_cate, id_cookie])
        y.append(t1)

    f.close()
    indices = [cookie2id, local2id, cate2id, word2id]
    with open(f_data, 'w') as f:
        cPickle.dump([x,y], f)
    with open(f_data_head, 'w') as f:
        cPickle.dump([x[:10000],y[:10000]], f)
    with open(f_indice, 'w') as f:
        cPickle.dump(indices, f)

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
    #process()
    #checkIndice()
    #load_data(f_data_head)
    # print sample2words(np.matrix([[3,4,5],[6,5,6]]))
    print sample2topwords(np.random.random_integers(0,10,size=(5,20,5)))
