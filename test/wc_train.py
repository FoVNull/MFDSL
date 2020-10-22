from gensim.models import word2vec
from glove import Glove
from glove import Corpus
from gensim.models import FastText
import logging


def word2v_train():
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'../wordvector/corpus/test_corpora_cut.txt')
    # sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

    model = word2vec.Word2Vec(sentences, sg=0, hs=1, window=5, min_count=3, size=100)

    # print(model.wv.vocab)

    # print(model.similarity("公平", "公开"))
    print(model.most_similar("悲哀"))
    # model.save('./reference/output')


def glove_train():
    sentences = word2vec.LineSentence(u'../wordvector/corpus/test_corpora_cut.txt')
    corpus_model = Corpus()
    corpus_model.fit(sentences, window=5)

    glove = Glove(no_components=100, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=5,
              no_threads=1, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    # glove.word_vectors[glove.dictionary['你']]
    print(glove.most_similar("悲哀", number=10))


def fasttext_train():
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'../wordvector/corpus/test_corpora_cut.txt')
    model = FastText(sentences,  size=100, window=5, min_count=1, iter=10, min_n=2, max_n=6, word_ngrams=0)
    for i in model.most_similar("悲哀"):
        print(i)
    # print(model.most_similar("悲哀"))


if __name__ == '__main__':
    word2v_train()
    glove_train()
    fasttext_train()