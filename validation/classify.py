from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import argparse
import pickle
import numpy as np


class Classifier:
    def __init__(self, args):
        self.args = args
        self.train_data = []
        self.model = SVC(kernel='linear', probability=True)

    def load_data(self):
        senti_vector = pickle.load(open(args.dic_path, 'rb'))
        with open("../corpus/" + self.args.corpus, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip().split("\t")) < 2:
                    continue
                data = line.split("\t")[0].split(" ")
                senti = line.strip().split("\t")[1]
                # if int(senti) < 3:
                #     senti = 'n'
                # else:
                #     if int(senti) > 3:
                #         senti = 'p'

                senten_vector = np.array([0.0] * self.args.dimension)
                count = 0
                for word in data:
                    if word not in senti_vector:
                        continue
                    senten_vector += np.array(senti_vector[word], dtype='float64')
                    count += 1
                if count == 0:
                    continue
                vector = senten_vector/count
                self.train_data.append((vector.tolist(), senti))

    def train(self, random_seed):
        x_train, x_test, y_train, y_test = train_test_split(
            [tp[0] for tp in self.train_data],
            [tp[1] for tp in self.train_data],
            test_size=0.5, random_state=random_seed, shuffle=True
        )
        self.model.fit(x_train, y_train)
        print("train acc:", self.model.score(x_train, y_train))
        return self.predict(x_test, y_test)

    def predict(self, x, y):
        predicted = self.model.predict(x)
        acc = accuracy_score(y, predicted)
        print("acc:", acc)
        evaluate_dimension = ["precision", "recall", "f1", "support"]

        # labels = ['p', 'n', '3']
        labels = ['p', 'n']
        # labels = ['1', '2', '3', '4', '5']
        eva = []
        for i, e in enumerate(precision_recall_fscore_support(y, predicted, labels=labels)):
            eva.append(e.tolist())
            print(evaluate_dimension[i], e.tolist())
        self.print_score(y, predicted, "macro")
        self.print_score(y, predicted, "micro")

        print(confusion_matrix(y, predicted, labels=labels))
        return eva, acc

    @staticmethod
    def print_score(y, predicted, average):
        print(average+" precision:", precision_score(y, predicted, average=average),
              " / "+average+" recall:", recall_score(y, predicted, average=average),
              " / "+average+" f1:", f1_score(y, predicted, average=average))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="sentiment classify validation")
    parse.add_argument("--corpus", type=str, default="classics/classics_test.tsv", help="specify corpus")
    parse.add_argument("--dic_path", type=str, default="../reference/output/wv.pkl",
                       help="specify sentiment dictionary")
    parse.add_argument("--dimension", default=100, type=int,
                       help="dimension of dictionary")
    parse.add_argument("--random_count", dest="count", default=10)

    args = parse.parse_args()

    classifier = Classifier(args)
    classifier.load_data()
    acc_sum = 0
    eva_sum = [[0, 0], [0, 0], [0, 0]]
    for seed in range(args.count):
        _eva, _acc = classifier.train(seed)
        for i in range(3):
            eva_sum[i][0] += _eva[i][0]
            eva_sum[i][1] += _eva[i][1]
        acc_sum += _acc
        print('#'*50)
    legend = ["precision", "recall", "f1"]
    for i in range(3):
        value = [eva_sum[i][0]/args.count, eva_sum[i][1]/args.count]
        print(legend[i], value, sum(value)/2)
    print(acc_sum/args.count)
