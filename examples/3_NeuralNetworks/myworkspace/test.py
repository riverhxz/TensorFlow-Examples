
from __future__ import print_function
import numpy as np
def cal_auc1(pred_label):
    assert pred_label.shape[1] > 1
    sz = pred_label.shape[0]

    b = pred_label.tolist()
    b = sorted(b,key=lambda x:x[0],reverse=True)
    sum=0
    pos = pred_label[:, 1].sum()
    neg = sz - pos
    for i in range(len(b)):
        sum += (i + 1) * b[i][1]
    auc = (sum - pos*(pos-1)/2)/(pos * neg)
    return auc

def cal_auc2(pred_label):
    assert pred_label.shape[1] > 1
    sz = pred_label.shape[0]
    pos = np.nonzero(pred_label[:, 1])[0].shape[0]
    neg = sz - pos
    #reverse ording indice
    indice = np.argsort(pred_label[:,0])
    rank = np.arange(1, sz + 1)
    # print (indice)
    # print(pred_label[:,1][indice])
    #sum of pos rank
    mean = np.dot(rank, pred_label[:,1][indice]/pos)

    # print("sum:{:f}, pos:{:f}, neg:{:f}".format(sum, pos, neg))
    auc = (mean - (pos+1)/2.)/ neg

    return auc

# def test_range():
#     x = np.array([[3, 1, 2,5,4],[0,1,0,1,0]]).transpose()
#     rank = np.arange(5,0,-1)
#
#     od =np.argsort(-x[:,0])
#     print(x[:, 1][od])
#     print (od, rank,)

if __name__ == '__main__':
    sz = 50000000
    # np.random.seed(47)
    a = np.random.random((sz,2)).astype("float32")
    a[:,1] = np.random.randint(0,2,(sz)).astype("float32")
    # a[:,0] = [3, 2, 14]
    # a[:,1] = [1, 1, 0]# a = np.random.random((sz,3)).astype("float16")
    print (cal_auc2(a))
    # test_range()