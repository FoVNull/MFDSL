from gensim.models import FastText
from gensim.models import word2vec
import logging
import argparse


def fasttext_train(tool):
    assert tool == 'fasttext' or tool == 'word2vec', 'you can choose: [word2vec, fasttext]'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'../corpus/zh_train.txt')
    if tool == 'fasttext':
        _model = FastText(sentences, size=300, iter=30, min_count=2, word_ngrams=3)
    else:
        _model = word2vec.Word2Vec(sentences, size=100, iter=10, min_count=2)
    # _model.save('../reference/wc_model/output')
    return _model


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="choose the way to train word vectors")
    parse.add_argument("--model", required=True, type=str, help="[word2vec, fasttext]")

    args = parse.parse_args()
    model = fasttext_train(args.model)
    import pickle

    # if args.model == 'fasttext':
    #     model = FastText.load("../reference/wc_model/output")
    # else:
    #     model = word2vec.Word2Vec.load("../reference/wc_model/output")
    dic = {}
    for i in model.wv.vocab:
        dic[i] = model.wv[i].tolist()

    pickle.dump(dic, open("../reference/output/wv.pkl", 'wb'))