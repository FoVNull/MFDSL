import jieba


def cut_words(path):
    stop_set = set(line.strip() for line in open("./reference/HIT_stopwords.txt", 'r', encoding='utf-8').readlines())
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            texts.append([word for word in jieba.lcut(line.strip(), use_paddle=True, cut_all=True)
                          if word not in stop_set])
    save_data("./corpus/test_corpora_cut.txt", texts)


def save_data(path, texts):
    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(" ".join(text) + "\n")


cut_words("./corpus/test_corpora.txt")
