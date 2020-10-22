from gensim.models import FastText
from gensim.models import word2vec
import logging
import argparse


def fasttext_train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'./corpus/test_corpora_cut.txt')
    model = FastText(sentences,  size=100, window=5, min_count=1, iter=10, min_n=2, max_n=6, word_ngrams=0)
    print(model.most_similar("公开"))


if __name__ == '__main__':
    fasttext_train()