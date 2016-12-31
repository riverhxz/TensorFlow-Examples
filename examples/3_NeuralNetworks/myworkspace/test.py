import numpy as np


def cal_auc(pred_label):
    assert pred_label.shape(1) == 2

    b = pred_label.tolist()
    b = sorted(b)
    print sorted(b, key=lambda x:x[0],reverse=True)
    sum=0
    pos = pred_label[:, 1].sum()
    neg = sz - pos
    for i in range(len(b)):
        sum += (i + 1) * b[i][1]

    auc = (sum - pos*(pos+1)/2)/(pos * neg)
    return auc

