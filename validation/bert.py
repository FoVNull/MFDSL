from bert_serving.client import BertClient
from validation.classify import Classifier
import argparse


class BertValidation(Classifier):
    def __init__(self, args):
        super().__init__(args)
        self.bc = BertClient()

    def load_data(self):
        texts = []
        with open("../corpus/amazon/book/all.txt", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                texts.append(line.strip())
        doc_vecs = self.bc.encode(texts)
        with open("../corpus/" + self.args.corpus, 'r', encoding='utf-8') as f:
            i = 0
            for line in f.readlines():
                if len(line.strip().split("\t")) < 2:
                    continue
                senti = line.strip().split("\t")[1]
                self.train_data.append((doc_vecs[i], senti))
                i += 1


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="sentiment classify validation")
    parse.add_argument("--corpus", type=str, default="amazon/book/vali2000.tsv", help="specify corpus")
    parse.add_argument("--random_count", dest="count", default=10)

    args = parse.parse_args()

    classifier = BertValidation(args)
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
    for i, lg in enumerate(legend):
        value = [eva_sum[i][0]/args.count, eva_sum[i][1]/args.count]
        print(lg, value, sum(value)/2)
    print(acc_sum/args.count)