from bare_model import Bare_Model

import argparse
import pickle
import numpy as np
import jieba

import tensorflow as tf
from transformers import XLNetTokenizer, BertTokenizer, MPNetTokenizer
from kashgari.embeddings import WordEmbedding, TransformerEmbedding
from kashgari.tasks.classification import BiLSTM_Model
from kashgari_local import XLNetEmbedding, MPNetEmbedding
from sklearn.model_selection import train_test_split


def _xlnet_corpus_gen(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            yield line.strip().split('\t')[0]


def read_amazon(path, senti_dic):
    x_data, y_data, features = [], [], []
    with open(path + '/vali6000.tsv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            k, v = line.strip().split('\t')
            line_features = np.array([])
            # load x
            x_data.append([word for word in k.split(" ") if word])
            # load features
            for word in k.split(" "):
                if word not in senti_dic.keys():
                    continue
                if len(line_features) > 0:
                    line_features += np.array(senti_dic[word])
                else:
                    line_features = np.array(senti_dic[word])
            # load y
            y_data.append(v)
            features.append(list(line_features / len(line_features)))
    return x_data, y_data, features


def read_hotel(senti_dic):
    pos_set, neg_set = [], []
    # 不去重
    with open('../../corpus/hotel/pos.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            pos_set.append(line.strip())
    with open('../../corpus/hotel/neg.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            neg_set.append(line.strip())

    x_data, y_data, features = [], [], []
    x_y_data = [(text, 'p') for text in pos_set][:3000] + [(text, 'n') for text in neg_set]
    np.random.shuffle(x_y_data)
    sv_dim = len(list(senti_dic.values())[0])
    for item in x_y_data:
        if not item[0]:
            continue
        x_data.append(item[0])
        y_data.append(item[1])
        line_features = []
        for word in jieba.cut(item[0], cut_all=True):
            # continue
            if word not in senti_dic.keys():
                line_features.append([0.]*sv_dim)
                continue
            if len(line_features) > 0:
                line_features.append(senti_dic[word])
            else:
                line_features.append(senti_dic[word])
        features.append(line_features)

    return x_data, y_data, features


class Trainer:
    def __init__(self, _args):
        self.model_type = _args.model_type
        if _args.model_type == "w2v":
            self.vocab_path = _args.model_folder + "/vocab.txt"
            self.vector_path = _args.model_folder + "/3C.vec"
        if _args.model_type == "bert":
            self.checkpoint_path = _args.model_folder + '/bert_model.ckpt'
            self.config_path = _args.model_folder + '/bert_config.json'
            self.vocab_path = _args.model_folder + '/vocab.txt'
        self.model_folder = _args.model_folder

    def _embedding(self, **params):
        if self.model_type == "w2v":
            embedding = WordEmbedding(self.vector_path)
        elif self.model_type in ["xlnet", 'xlnet_zh']:
            # 输入语料用于构建xlnet词表，生成器形式
            embedding = XLNetEmbedding(self.model_folder, _xlnet_corpus_gen(params["xlnet_corpus"]))
        elif self.model_type == "mpnet":
            embedding = MPNetEmbedding(self.model_folder)
        else:
            embedding = TransformerEmbedding(self.vocab_path, self.config_path, self.checkpoint_path,
                                             model_type=self.model_type)

        return embedding

    def _tokenizing(self, _x):
        sentences_tokenized = _x

        if self.model_type == 'bert':
            tokenizer = BertTokenizer.load_from_vocab_file(self.vocab_path)
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]
        if self.model_type == 'xlnet':
            tokenizer = XLNetTokenizer(self.model_folder + "/spiece.model")
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]
        if self.model_type == 'mpnet':
            tokenizer = MPNetTokenizer(self.model_folder + '/vocab.txt')
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]
        if self.model_type == 'xlnet_zh':
            tokenizer = XLNetTokenizer(self.model_folder + "/spiece.model")
            sentences_tokenized = [tokenizer.tokenize(seq) for seq in _x]

        return sentences_tokenized

    def train(self, xlnet_corpus=None, eval=True):
        # x_data, y_data, features = read_amazon('/home/sz/project/MFDSL/corpus/amazon/book',
        #                                        pickle.load(open('../../reference/output/sv.pkl', 'rb')))
        x_data, y_data, features = read_hotel(pickle.load(open('../../reference/output/sv.pkl', 'rb')))

        if self.model_type in ['xlnet', 'xlnet_zh']:
            x_data = self._tokenizing(x_data)

        embedding = self._embedding(xlnet_corpus=xlnet_corpus)

        data_gen = self.k_fold(6, x_data, y_data, features)
        reports = []
        for x_test, y_test, test_features, \
            x_train, y_train, train_features in data_gen:
            model = Bare_Model(embedding, feature_D=len(train_features[0][0]))
            # model = BiLSTM_Model(embedding)

            print("train-{}, test-{}".format(len(x_train), len(x_test)))
            with tf.device('/gpu:2'):
                model.fit(x_train=(x_train, train_features), y_train=y_train,
                          batch_size=32, epochs=15, callbacks=None, fit_kwargs=None)
                # model.fit(x_train=x_train, y_train=y_train,
                #           batch_size=32, epochs=10, callbacks=None, fit_kwargs=None)
            with tf.device('/gpu:3'):
                reports.append(self.evaluate(model, x_test, y_test, test_features))
                # reports.append(self.evaluate_nonf(model, x_test, y_test))
        self.save_res(reports, 6)

        # 测试
        # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1000, random_state=114514)
        # # x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=0.5)
        # model = BiLSTM_Model(embedding=embedding)
        # with tf.device('/gpu:2'):
        #     model.fit(x_train, y_train,
        #               # x_dev, y_dev,
        #               batch_size=32, epochs=10)
        # with tf.device('/gpu:3'):
        #     model.evaluate(x_test, y_test, batch_size=32)

        # x_train, x_test, y_train, y_test, train_features, test_features = train_test_split(
        #     x_data, y_data, features, test_size=1000, random_state=114514)
        # # x_vali, x_test, y_vali, y_test, vali_features, test_features = train_test_split(
        # #     x_test, y_test, test_features, train_size=500)
        # model = Bare_Model(embedding, feature_D=len(train_features[0][0]))
        # with tf.device('/gpu:2'):
        #     model.fit(x_train=(x_train, train_features), y_train=y_train,
        #               # x_validate=(x_vali, vali_features), y_validate=y_vali,
        #               batch_size=32, epochs=15, callbacks=None, fit_kwargs=None)
        # with tf.device('/gpu:3'):
        #     self.evaluate(model, x_test, y_test, test_features)

    @staticmethod
    def evaluate(model, x_test_pure, y_test, features):
        # y_test转置，便于model.evaluate处理多任务输出
        y_test = list(map(lambda x: list(x), zip(*y_test)))
        reports = model.evaluate((x_test_pure, features), y_test, batch_size=32)
        # for report in reports:
        #     print(report)
        # model.save("./models/output")
        return reports

    @staticmethod
    def evaluate_nonf(model, x_test_pure, y_test):
        reports = model.evaluate(x_test_pure, y_test, batch_size=32)
        return reports

    @staticmethod
    def k_fold(k, x, y, f):
        # part = math.ceil((len(x) / k))
        part = 1000
        for i in range(0, len(x), part):
            yield x[i:i + part], y[i:i + part], f[i:i + part], \
                  x[:i] + x[i + part:], y[:i] + y[i + part:], f[:i] + f[i + part:]

    @staticmethod
    def save_res(reports, k):
        with open("./reports.tsv", 'w', encoding='utf-8') as f:
            acc, p, r, f1, auc = [], [], [], [], []
            for report in reports:
                acc.append(report[0]['detail']['accuracy'])
                p.append(report[0]['detail']['macro avg']['precision'])
                r.append(report[0]['detail']['macro avg']['recall'])
                f1.append(report[0]['detail']['macro avg']['f1-score'])
                auc.append(report[0]['auc'])
            res = "task{};{}-Fold\n" \
                  "avg-macro:acc={}; p={}; r={}; f1={}; auc{}\n" \
                  "max-macro:acc={}; p={}; r={}; f1={}; auc{}\n" \
                  "min-macro:acc={}; p={}; r={}; f1={}; auc{}\n" \
                .format(0, k,
                        sum(acc) / k, sum(p) / k, sum(r) / k, sum(f1) / k, sum(auc) / k,
                        max(acc), max(p), max(r), max(f1), max(auc), min(acc), min(p), min(r), min(f1), min(auc))
            f.write(res)
            f.write(f'std:{np.std(f1)}')
        pickle.dump((acc, p, r, f1, auc), open("./details.pkl", 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training model")

    parser.add_argument("--model_folder", type=str, help="universal pretrained model path")
    parser.add_argument("--model_type", type=str, default='xlnet_zh',
                        help="transfer model type, [bert, w2v, xlnet, xlnet_zh, mpnet]")
    args = parser.parse_args()
    trainer = Trainer(args)

    # trainer.train(xlnet_corpus='/home/sz/project/MFDSL/corpus/amazon/book/all.txt')
    trainer.train(xlnet_corpus='/home/sz/project/MFDSL/corpus/hotel/all.txt')
