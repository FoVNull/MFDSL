from gensim.models import FastText
from gensim.models import word2vec
import logging
import argparse


def fasttext_train(tool):
    assert tool == 'fasttext' or tool == 'word2vec', 'you can choose: [word2vec, fasttext]'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'../corpus/hotel_en/all4train.txt')
    if tool == 'fasttext':
        model = FastText(sentences, size=100, window=5, min_count=2, iter=30, min_n=2, max_n=8, word_ngrams=0)
    else:
        model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=1, iter=30)
    model.save('../reference/wc_model/output')


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="choose the way to train word vectors")
    parse.add_argument("--model", required=True, type=str, help="[word2vec, fasttext]")

    args = parse.parse_args()
    fasttext_train(args.model)
    import pickle

    model = FastText.load("../reference/wc_model/output")
    dic = {}
    for i in model.wv.vocab:
        dic[i] = model.wv[i].tolist()

    pickle.dump(dic, open("../reference/output/ft.pkl", 'wb'))