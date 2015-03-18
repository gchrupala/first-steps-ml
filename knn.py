from knn_functions import *


def parse(line):
    fields = line.split()
    x = [ float(i) for i in fields[:-1] ]
    y = fields[-1]
    return (x,y)

def print_evaluation(X_tr, Y_tr, X_de, Y_de):
    print "{:>3s} {:>3s} {:>3s} {:>7s}".format("k", "err", "N", "err/N")
    for k in range(1,16):
        model = train(X_tr, Y_tr)
        predicted = [ predict(model, x, k=k) for x in X_de ]
        err, N, rate = error_rate(Y_de, predicted)
        print "{:3d} {:3d} {:3d} {:7.3f}".format(k, err, N, rate)

def main():
    def select(xy):
        x, y = xy
        return (x[0:2], y)
    X_tr, Y_tr = zip(*[ select(parse(line)) for line in open('iris-train.txt') ])
    X_de, Y_de = zip(*[ select(parse(line)) for line in open('iris-dev.txt') ])
    print "Using first two features"
    print_evaluation(X_tr, Y_tr, X_de, Y_de)
    print "Using all four features"
    X_tr, Y_tr = zip(*[ parse(line) for line in open('iris-train.txt') ])
    X_de, Y_de = zip(*[ parse(line) for line in open('iris-dev.txt') ])
    print_evaluation(X_tr, Y_tr, X_de, Y_de)

main()



