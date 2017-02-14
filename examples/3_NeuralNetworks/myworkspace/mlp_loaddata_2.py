# -*- coding: utf-8 -*-
import numpy as np
import sys
# from six.moves import cPickle
import h5py as h5
import codecs

f_data = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata/xx_sample.h5'
f_indice = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata.pkl'
f_data_head = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata/xx_head.h5'

data_file="/tmp/data/np/sample.h5"
encoding = 'utf-8'
class SampleTransform(object):

    def load_data(self, path=data_file, nb_words=None, skip_top=0,
                  maxlen=None, test_split=0.2, seed=113,
                  end_char=1, oov_char=2, index_from=3):
        data = h5.File(path)
        return data

    def __init__(self, input_fn='/home/hehehe/data/app.1m', max_seq_lenth=20, output_dir='/home/hehehe/data/', output_fn='sample.h5', start=0 ,end=None):
        self.input_fn = input_fn
        self.max_seq_lenth = max_seq_lenth

        self.start = start
        self.end = end
        self.num_sample = end - start if end != None else None
        self.output_dir = output_dir
        self.output_fn = output_fn
        self.internal_id = 0
        self.int_type = "int32"
        self.float_type = "float32"
        self.line_num = -1
    def process(self, ):
        self.internal_id = 0
        seed = 47
        f = codecs.open(self.input_fn,'r', encoding)
        start_of_word = 3

        y = np.zeros(self.num_sample, self.float_type)
        cookie_set = np.zeros(self.num_sample, dtype=self.int_type)
        local_set = np.zeros(self.num_sample, dtype=self.int_type)
        cate_set = np.zeros(self.num_sample, dtype=self.int_type)
        word_set = np.zeros((self.num_sample, self.max_seq_lenth), dtype=self.int_type)
        word_len_set = np.zeros(self.num_sample, dtype=self.int_type)
        conf = {"y":y, "cookie":cookie_set, "local":local_set, "cate":cate_set, "word_len":word_len_set}
        df = h5.File(self.output_dir + '/' + self.output_fn, "w")

        def shuffle(x):
            np.random.seed(seed)
            np.random.shuffle(x)

        def save(data, name):
            # d = np.asarray(data,dtype="int32")
            data=data[:self.internal_id]
            shuffle(data)
            df.create_dataset(data=data, name=name)

        def collect(t5):
            click, cookie, local, cate, word = t5
            y[self.internal_id] = click
            cookie_set[self.internal_id] = cookie
            local_set[self.internal_id] = local
            cate_set[self.internal_id] = cate
            lenx = min(len(word),self.max_seq_lenth)
            word_set[self.internal_id, :lenx] = word[:lenx]
            word_len_set[self.internal_id] = lenx

            self.internal_id += 1

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
            if self.num_sample is not None and self.num_sample == self.internal_id:
                break
            self.line_num += 1
            if self.start > self.line_num:
                continue

            t5 = parseText(line.strip())
            if t5 == None:
                continue
            collect(t5)

        map(lambda (k, v) : save(v, k), conf.iteritems())

        wordmx = np.zeros((self.internal_id, self.max_seq_lenth),"int32")
        for i in range(len(word_set)):
            ll = min(self.max_seq_lenth, len(word_set[i]))
            wordmx[i,:ll] = word_set[i][:ll]

        shuffle(wordmx)
        df.create_dataset("word",data=wordmx)
        print("sample num:{:d}".format(len(y)))
        df.close()

if __name__ == '__main__':
    #process(input_fn='/home/hehehe/data/data_1d', limit=50000000,  output_fn='sample.50m.h5')
    SampleTransform(input_fn='/home/hehehe/data/data_1d', start=0, end=25000000,  output_fn='sample.25m.h5').process()