from gensim.models import word2vec
import logging


def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'./corpus/test_corpora_cut.txt')
    # sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

    model = word2vec.Word2Vec(sentences, sg=0, hs=1, window=5, min_count=3, size=100)

    # print(model.wv.vocab)

    # print(model.similarity("公平", "公开"))
    # print(model.most_similar("好"))
    model.save('./reference/output')


if __name__ == '__main__':
    train()