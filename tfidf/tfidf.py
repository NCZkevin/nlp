import jieba
import jieba.posseg as psg
import math
import numpy as np
import functools

def get_stopword_list():
    stop_word_path ='./stopword.txt'
    stopword_list = [sw.replace('\n','') for sw in open(stop_word_path).readlines()]
    return stopword_list

def seg_to_list(sentence, pos=False):
    if not pos:
        seg_list = jieba.cut(sentence)
    else:
        seg_list = psg.cut(sentence)
    return seg_list

def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []

    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        if not word in stopword_list and len(word)>1:
            filter_list.append(word)
    
    return filter_list

def load_data(pos=False, data_path='./data.txt'):
    doc_list = []
    for line in open(data_path,'r'):
        content = line.strip()
        seg_list = seg_to_list(content,pos)
        filter_list = word_filter(seg_list,pos)
        doc_list.append(filter_list)

    return doc_list

def train_idf(doc_list):
    idf_dic = {}

    tf_count = len(idf_dic)

    #每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
    
    for k,v in idf_dic.items():
        idf_dic[k] = math.log(tf_count / (1.0+v))

    default_idf = math.log(tf_count / (1.0))
    return idf_dic,default_idf

def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1

# TF-IDF类
class TFIDF(object):

    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num
    
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        
        tt_count = len(self.word_list)
        for k,v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count

        return tf_dic

    def get_tfidf(self):
        tdidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tdidf_dic[word] = tfidf

        for k,v in sorted(tdidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + ':' + v,end='')

# 调用算法
def tfidf_extract(word_list, pos=False, keyword_num=5):
    
    doc_list = load_data(pos)
    idf_dic, default_idf= train_idf(doc_list)
    tfidf_model = TFIDF(idf_dic, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf()

if __name__ == '__main__':

    text = ''
    pos = False
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)

    print("TF-IDF模型结果为：")
    tfidf_extract(filter_list)