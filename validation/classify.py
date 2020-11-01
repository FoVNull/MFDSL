from sklearn.svm import SVC
import argparse
import pickle
import numpy as np


class Classifier:
    def __init__(self, args):
        self.args = args
        self.train_data = []
        self.pos = []
        self.neg = []
        self.load_data()
        self.model = SVC(kernel='linear', probability=True)

    def load_data(self):
        senti_vector = pickle.load(open(args.dic_path, 'rb'))
        with open("../corpus/" + self.args.corpus+"/train_cut.tsv", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip().split("\t")) < 2:
                    continue
                data = line.split("\t")[0].split(" ")
                senti = line.strip().split("\t")[1]

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

        def load_test(path):
            res = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    senten_vector = np.array([0.0] * self.args.dimension)
                    data = line.split(" ")
                    count = 0
                    for word in data:
                        if word not in senti_vector:
                            continue
                        senten_vector += np.array(senti_vector[word], dtype='float64')
                        count += 1
                    if count == 0:
                        continue
                    vector = senten_vector/count
                    res.append(vector.tolist())
            return res
        self.pos = load_test("../corpus/hotel/pos_test_cut.txt")
        self.neg = load_test("../corpus/hotel/neg_test_cut.txt")

    def train(self):
        x = [tp[0] for tp in self.train_data]
        y = [tp[1] for tp in self.train_data]
        self.model.fit(x, y)
        print("train acc:", self.model.score(x, y))
        self.predict()

    def predict(self):
        pos_res = self.model.predict(self.pos)
        pos_dic = {}
        for i in pos_res:
            pos_dic[i] = pos_dic.get(i, 0)+1

        neg_res = self.model.predict(self.neg)
        neg_dic = {}
        for i in neg_res:
            neg_dic[i] = neg_dic.get(i, 0)+1

        p_cor = pos_dic['p']
        p_wro = pos_dic['n']
        n_cor = neg_dic['n']
        n_wro = neg_dic['p']

        p_pre = p_cor/(p_cor + p_wro)
        p_rec = p_cor/(p_cor + n_wro)
        print(">pos predict:\n", "p:", p_cor, " n:", p_wro, "\npos precision:", p_pre,
              "\nrecall:", p_rec, "\nf1:", 2*p_pre*p_rec/(p_pre+p_rec))
        print("#" * 40)

        n_pre = n_cor/(n_cor + n_wro)
        n_rec = n_cor/(n_cor + p_wro)
        print(">neg predict:\n", "p:", n_wro, " n:", n_cor, "\nneg precision:", n_pre,
              "\nrecall:", n_rec, "\nf1:", 2*n_pre*n_rec/(n_pre+n_rec))
        print("#" * 40)
        print("overall acc: ", (p_cor+n_cor) / (p_cor + p_wro + n_cor + n_wro))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="sentiment classify validation")
    parse.add_argument("--corpus", type=str, default="hotel", help="specify corpus")
    parse.add_argument("--dic_path", type=str, default="../sv.pkl", help="specify sentiment dictionary")
    parse.add_argument("--dimension", default=200, type=int,
                       help="dimension of dictionary")

    args = parse.parse_args()

    classifier = Classifier(args)
    classifier.train()
