# -*- coding: utf-8 -*-

from __future__ import division
import sys
import random
from perceptron_functions import *


def prepare_data():
    train = []
    with open("sentiment.feat") as inp:
        for line in inp:
            xy = parse_line(line)
            train.append(xy)
    indexes = range(0, len(train))
    SEED = 4096
    random.seed(SEED)
    random.shuffle(indexes)
    XY = [ (train[i][0], 1 if train[i][1] > 5 else -1) for i in indexes ]
    return XY

def evaluate(gold, predicted):
    N = len(gold)
    errs = sum([ 1 if p != y else 0 for p,y in zip(predicted, gold)])
    return (errs, N, errs/N)

def main():
    iterations = 20
    XY = prepare_data()
    XY_train = XY[:5000]
    XY_dev   = XY[5000:]
    Y_train = [ xy[1] for xy in XY_train ]
    Y_dev   = [ xy[1] for xy in XY_dev ]
    model = initialize()
    sys.stdout.write("{:>3s} {:>7s} {:>7s}\n".
                     format("Iter", "err_tr", "err_dev"))
    for i in range(1,iterations+1):
        predicted_train = learn(model, XY_train)
        _, _, rate_train = evaluate(Y_train, predicted_train)
        predicted_dev = [ predict(model, x) for (x,_) in XY_dev ]
        _, _, rate_dev = evaluate(Y_dev, predicted_dev)
        sys.stdout.write("{:3d} {:7.3f} {:7.3f}\n".
                         format(i, rate_train, rate_dev))

if __name__ == '__main__':
    main()
