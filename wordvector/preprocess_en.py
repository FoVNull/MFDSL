import en_core_web_lg
import pandas as pd
from lxml import etree
import re

word_flag = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'VERB']  # , 'NOUN', 'VERB', 'INTJ']
nlp = en_core_web_lg.load()


def cut_xlsx(path):
    df = pd.read_excel(path, header=0)
    texts = [content for content in df['内容'] if isinstance(content, str)]

    # rates = df['星数'].values.reshape(-1).tolist()

    def rate2polarity(r):
        if r >= 3:
            return "p"
        else:
            return "n"

    rates = [rate2polarity(int(rate)) for i, rate in enumerate(df['评级']) if isinstance(df['内容'][i], str)]

    cut_text = []
    for doc in nlp.pipe(texts):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        cut_text.append(s)
    p = 0; n = 0
    texts_test = []
    rates_test = []
    for _ in range(len(rates)):
        if not cut_text[_]:
            continue
        if rates[_] == 'p' and p < 600:
            texts_test.append(cut_text[_])
            rates_test.append('p')
            p += 1
        if rates[_] == 'n' and n < 600:
            texts_test.append(cut_text[_])
            rates_test.append('n')
            n += 1

    with open("../corpus/classics/classics.txt", 'w', encoding='utf-8') as f:
        for _ in texts:
            f.write(_ + "\n")
    save_data("../corpus/classics/classics_cut.txt", cut_text)
    save_data_rates("../corpus/classics/classics_test.tsv", texts_test, rates_test)


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


def merge(path1, path2, save_path, save_path2):
    content = [(line.strip(), 'p') for line in open(path1, 'r', encoding='utf-8').readlines()] + \
              [(line.strip(), 'n') for line in open(path2, 'r', encoding='utf-8').readlines()]
    with open(save_path, 'w', encoding='utf-8') as f:
        for c in content:
            f.write(c[0] + "\t" + c[1] + "\n")

    with open(save_path2, 'w', encoding='utf-8') as f:
        for c in content[:1000] + content[-1000:]:
            f.write(c[0] + "\t" + c[1] + "\n")


# merge("../corpus/amazon/book/review_positive_cut.txt", "../corpus/amazon/book/review_negative_cut.txt", "../corpus/amazon/book/vali6000.tsv")
# cut_xlsx("../corpus/classics/classics_en.xlsx")
# cut_txt("../corpus/book/book_review.txt")
# cut_txt("../corpus/NYT_comment.txt")
# cut_xml("../corpus/classics/classics.xml")


# for domain in ['dvd', 'electronics', 'kitchen', 'video']:
#     print(domain)
#     # cut_txt("../corpus/amazon/"+domain+"/all.txt")
#     cut_txt("../corpus/amazon/"+domain+"/review_positive.txt")
#     cut_txt("../corpus/amazon/"+domain+"/review_negative.txt")
#     merge("../corpus/amazon/"+domain+"/review_positive_cut.txt",
#           "../corpus/amazon/"+domain+"/review_negative_cut.txt", "../corpus/amazon/"+domain+"/vali6000.tsv",
#           "../corpus/amazon/"+domain+"/vali2000.tsv")

cut_xlsx("../corpus/classics/amazon_reviews_gbk.xlsx")
