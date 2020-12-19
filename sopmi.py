import jieba.posseg as pseg
import jieba
import math


class SoPmi:
    def __init__(self, path, seeds):
        self.seg_data = [line.split("\t")[0].split(" ") for line in open(path, 'r', encoding='utf-8')]
        self.sentiment_path = seeds
        self.cowords_list = self.collect_cowords(self.sentiment_path, self.seg_data)
        self.word_dict, self.all = self.collect_worddict(self.seg_data)
        self.co_dict = self.collect_cowordsdict(self.cowords_list)

    '''统计搭配次数'''
    def collect_cowords(self, sentiment_path, seg_data):
        def check_words(sent):
            if set(sentiment_words).intersection(set(sent)):
                return True
            else:
                return False

        cowords_list = list()
        window_size = 10
        count = 0
        sentiment_words = [line.strip().split('\t')[0] for line in open(sentiment_path)]
        for sent in seg_data:
            count += 1
            if check_words(sent):
                for index, word in enumerate(sent):
                    if index < window_size:
                        left = sent[:index]
                    else:
                        left = sent[index - window_size: index]
                    if index + window_size > len(sent):
                        right = sent[index + 1:]
                    else:
                        right = sent[index: index + window_size + 1]
                    context = left + right + [word]
                    if check_words(context):
                        for index_pre in range(0, len(context)):
                            if check_words([context[index_pre]]):
                                for index_post in range(index_pre + 1, len(context)):
                                    cowords_list.append(context[index_pre] + '@' + context[index_post])
        return cowords_list

    '''统计词频'''
    @staticmethod
    def collect_worddict(seg_data):
        word_dict = dict()
        _all = 0
        for line in seg_data:
            for word in line:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        _all = sum(word_dict.values())
        return word_dict, _all

    '''统计词共现次数'''
    @staticmethod
    def collect_cowordsdict(cowords_list):
        co_dict = dict()
        for co_words in cowords_list:
            if co_words not in co_dict:
                co_dict[co_words] = 1
            else:
                co_dict[co_words] += 1
        return co_dict

    '''计算So-Pmi值'''
    def collect_candiwords(self, senti_word, candi_word):
        """互信息计算公式"""
        def compute_mi(p1, p2, p12):
            return math.log2(p12) - math.log2(p1) - math.log2(p2)

        '''计算sopmi值'''
        def compute_sopmi(candi_word, senti_word, word_dict, co_dict, all):
            _value = 0.0
            p1 = word_dict[senti_word] / all
            p2 = word_dict[candi_word] / all
            pair = senti_word + '@' + candi_word
            if pair in co_dict:
                p12 = co_dict[pair] / all
                _value += compute_mi(p1, p2, p12)
            return _value

        pmi_dict = compute_sopmi(candi_word, senti_word, self.word_dict, self.co_dict, self.all)
        return pmi_dict

    def sopmi(self, candi_word, senti_word):
        sopmi_value = self.collect_candiwords(senti_word, candi_word)
        return sopmi_value


sp = SoPmi("./corpus/hotel/all_cut.tsv", "./reference/output/seeds.tsv")
# sp.sopmi("整洁", "最差")
# sp.sopmi("整洁", "干净")