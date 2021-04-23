import en_core_web_lg
import pandas as pd
from lxml import etree
import re

word_flag = ['ADJ', 'ADV', 'INTJ']  # , 'NOUN', 'VERB', 'INTJ']
nlp = en_core_web_lg.load()


def cut_xlsx(path):
    df = pd.read_excel(path, header=0)
    texts = df['内容'].values.reshape(-1).tolist()
    rates = df['星数'].values.reshape(-1).tolist()
    cut_text = []
    for doc in nlp.pipe(texts):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        cut_text.append(s)

    save_data_rates(path[:-5] + "_cut.txt", cut_text, rates)


def cut_xml(path):
    tree = etree.parse(path)
    root = tree.getroot()
    comments = root.xpath("//summary_trans")
    texts = [comment.text.strip() for comment in comments if comment.text is not None] + \
            [comment.text.strip() for comment in root.xpath("//text_trans") if comment.text is not None]

    cut_text = []
    for doc in nlp.pipe(texts):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        cut_text.append(s)

    save_data(path[:-4] + "_cut.txt", cut_text)


def cut_txt(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            texts.append(line.strip())
    cut_text = []
    for doc in nlp.pipe(texts):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        cut_text.append(s)

    save_data(path[:-4] + "_cut.txt", cut_text)


def save_data(path, texts):
    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            if not text:
                continue
            f.write(" ".join(text) + "\n")


def save_data_rates(path, texts, rates):
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(texts)):
            if not texts[i]:
                continue
            f.write(" ".join(texts[i]) + "\t" + str(rates[i]) + "\n")


def merge(path1, path2, save_path):
    content = [(line.strip(), 'p') for line in open(path1, 'r', encoding='utf-8').readlines()] + \
              [(line.strip(), 'n') for line in open(path2, 'r', encoding='utf-8').readlines()]
    with open(save_path, 'w', encoding='utf-8') as f:
        for c in content:
            f.write(c[0] + "\t" + c[1] + "\n")


# cut_txt("../corpus/amazon/book/book_unlabeled.txt")
# cut_txt("../corpus/amazon/book/review_positive.txt")
# cut_txt("../corpus/amazon/book/review_negative.txt")
# merge("../corpus/amazon/book/review_positive_cut.txt", "../corpus/amazon/book/review_negative_cut.txt", "../corpus/amazon/book/vali6000.tsv")
# cut_xlsx("../corpus/classics/classics_en.xlsx")
# cut_txt("../corpus/book/book_review.txt")
# cut_txt("../corpus/NYT_comment.txt")
# cut_xml("../corpus/classics/classics.xml")


data = pd.read_excel("../corpus/classics/amazon_reviews_sz_gbk.xlsx", header=0)
tagger = en_core_web_lg.load()
with open("../corpus/classics/classics_reviews.txt", 'w', encoding='utf-8') as f:
    for text in data['内容']:
        if str(text) == 'nan':
            continue
        tokens = [token.text
                  for token in tagger(str(text))
                  if not tagger.vocab[token.text].is_stop and token.pos_ in word_flag
                  ]
        if len(tokens) == 0:
            continue
        f.write(" ".join(tokens) + "\n")

# cut_csv(pos, 'p', "../corpus/hotel_en/pos.txt")
# cut_csv(neg, 'p', "../corpus/hotel_en/neg.txt")
# cut_csv(neg[:8000], 'n', "../corpus/hotel_en/validation10000.tsv")
# cut_csv(pos[:8000], 'p', "../corpus/hotel_en/pos_cut.tsv")
# save(pos+neg, 'n', "../corpus/hotel_en/all4train.txt")
