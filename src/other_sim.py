from numpy import dot
from numpy.linalg import norm
from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.cluster import AgglomerativeClustering
from scipy import spatial
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
import collections
from numpy import dot
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

class EmbeddingMethod:
    glove = "glove"
    bert = "bert"
    fasttext = "fasttext"

class ClusterMethod:
    sentence = "sentence"
    fact = "fact"

def load_bert_model(path):
    word_embedding_model = models.Transformer(path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

def elu_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

def load_glove_vectors(glove_file="/home/yaguang/pretrained_models/glove.6B.50d.txt"):
    """Load the glove word vectors"""
    word_vectors = {}
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = [float(x) for x in split[1:]]
    return word_vectors

def load_fasttext_vectors(ft_file="/home/yaguang/pretrained_models/wiki-news-300d-1M.vec"):
    """Load the fasttext vectors"""
    word_vectors = {}
    with open(ft_file) as f:
        f.readline()
        for line in f:
            split = line.split()
            word_vectors[split[0]] = [float(x) for x in split[1:]]
    return word_vectors

def filter_tokens(sentence):
    tmp = []
    exclude = set(string.punctuation)
    for t in word_tokenize(sentence):
        t = t.strip().lower()
        t = ''.join(ch for ch in t if ch not in exclude)
        if not t:
            continue
        #if t in stop_words:
         #   continue
        tmp.append(t)
    return tmp

def get_word_embedding(word, emb_size, model):
    return model[word] if word in model else list(np.random.uniform(-0.25, 0.25, emb_size))

def get_bert_embeddings(sentences, model):
    sentence_embeddings = model.encode(sentences).tolist()
    return sentence_embeddings

def tokens_to_embeddings(tokens, model):
    if embedding_method == EmbeddingMethod.glove:
        #return np.mean([get_word_embedding(token, 50, model) for token in tokens], 0)
        return np.sum([get_word_embedding(token, 50, model) for token in tokens], 0)

    if embedding_method == EmbeddingMethod.fasttext:
        return np.mean([get_word_embedding(token, 300, model) for token in tokens], 0)

def get_embeddings(filename):
    df = pd.read_csv(filename).drop_duplicates()
    chapter_dic, section_dic, article_dic = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    recitals = []
    texts = []
    count = 0
    df['recital'] = df['recital'].str.lower()
    for index, row in df.iterrows():
        chapter, section, article, recital = row['chapter'], row['section'], row['article'], row['recital']
        if not recital:
            continue
        if recital == "undefined" :
            continue
        if article  != "undefined" :
            article_dic[article].append(count)
        tokens = filter_tokens(recital)
        if len(tokens) <= 1:
            continue
        recitals.append(recital)
        count += 1
    return article_dic, recitals

def cos_sim(a,b):
    if norm(a) == 0:
        print ("a is ")
        print (a)
    if norm(b) == 0:
        print ("b is ")
        print (b)
    return dot(a, b)/(norm(a)*norm(b))

def sentence_process(filename, gdpr_embeddings, bdpr_embeddings):
    w = open(filename, 'w')
    w.write("\t".join(["gdpr recital", "bdpr recital", "similarity"])+"\n")
    for i in range(len(gdpr_embeddings)):
        temp = []
        for j in range(len(bdpr_embeddings)):
            #sim = 1 - spatial.distance.cosine(gdpr_embeddings[i], bdpr_embeddings[j])
            sim = cos_sim(gdpr_embeddings[i], bdpr_embeddings[j])
            temp.append((sim, i, j))
        temp.sort(reverse=True)
        temp = temp[:30]
        for val in temp:
            w.write("\t".join([gdpr_text[val[1]], bdpr_text[val[2]], str(val[0])])+"\n")
    w.close()

def article_process(filename, gdpr_article_dic, bdpr_article_dic, gdpr_embeddings, bdpr_embeddings):
    w = open(filename, 'w')
    w.write("\t".join(["gdpr article", "bdpr article", "similarity"])+"\n")
    for key2 in gdpr_article_dic:
        temp = []
        for key1 in bdpr_article_dic:
            b_idx = bdpr_article_dic[key1]
            b_texts = ".".join([key1]+[bdpr_text[idx] for idx in b_idx])
            b_sum = np.sum([bdpr_embeddings[idx] for idx in b_idx], 0)

            g_idx = gdpr_article_dic[key2]
            g_texts = ".".join([key2]+[gdpr_text[idx] for idx in g_idx])
            g_sum = np.sum([gdpr_embeddings[idx] for idx in g_idx], 0)

            sim = 1 - spatial.distance.cosine(b_sum, g_sum)
            temp.append((sim, g_texts, b_texts))
        temp.sort(reverse=True)
        temp = temp[:30]
        for val in temp:
            w.write("\t".join([val[1], val[2], str(val[0])])+"\n")
    w.close()


chapter_gdpr, section_gdpr, article_gdpr = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)

glove_model = load_glove_vectors()
siamese_model =  SentenceTransformer("bert-base-nli-mean-tokens")
siamese_model.max_seq_length = 512
bert_model = load_bert_model("/home/yaguang/pretrained_models/uncased_L-12_H-768_A-12")
bert_model.max_seq_length = 512

bdpr_article_dic, bdpr_text = get_embeddings("data/LGPD-ES-Brazil-converted.csv")
gdpr_article_dic, gdpr_text = get_embeddings("data/GDPR-EN-Europe-converted.csv")

vectornizer = TfidfVectorizer()
trans_data = vectornizer.fit_transform(gdpr_text+bdpr_text).toarray()
gdpr_embeddings = trans_data[:len(gdpr_text)]
bdpr_embeddings = trans_data[len(gdpr_text):]

sentence_process("simi_sentence_tfidf.csv", gdpr_embeddings, bdpr_embeddings)
article_process("simi_article_tfidf.csv", gdpr_article_dic, bdpr_article_dic, gdpr_embeddings, bdpr_embeddings)
