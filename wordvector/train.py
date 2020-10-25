from gensim.models import FastText
from gensim.models import word2vec
import logging
import argparse


def fasttext_train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'../corpus/all_cut.txt')
    # model = FastText(sentences, size=200, window=5, min_count=2, iter=20, min_n=2, max_n=6, word_ngrams=0)
    model = word2vec.Word2Vec(sentences, size=200, window=5, min_count=1, iter=20)
    model.save('../reference/wc_model/output')


if __name__ == '__main__':
    fasttext_train()
