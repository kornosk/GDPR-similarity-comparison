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
    for t in tknzr.tokenize(sentence):
        t = t.strip().lower()
        t = ''.join(ch for ch in t if ch not in exclude)
        t = lemmatizer.lemmatize(t)
        if not t:
            continue
        if t in stop_words:
            continue
        tmp.append(t)
    return tmp

def get_word_embedding(word, emb_size, model):
    return model[word] if word in model else list(np.random.uniform(0,0, emb_size))

def get_bert_embeddings(sentences, model):
    sentence_embeddings = model.encode(sentences).tolist()
    return sentence_embeddings

def tokens_to_embeddings(tokens, model):
    if embedding_method == EmbeddingMethod.glove:
        return np.mean([get_word_embedding(token, 50, model) for token in tokens], 0)
    if embedding_method == EmbeddingMethod.fasttext:
        return np.mean([get_word_embedding(token, 300, model) for token in tokens], 0)

def get_embeddings(model, filename):
    df = pd.read_csv(filename).drop_duplicates()
    chapter_dic, section_dic, article_dic = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    recitals = []
    texts = []
    glove_embeddings = []
    f = open(filename)
    count = 0
    df['recital'] = df['recital'].str.lower()
    for index, row in df.iterrows():
        #fact = fact.encode('ascii', errors='ignore').decode().replace("\n", ". ")
        chapter, section, article, recital = row['chapter'], row['section'], row['article'], row['recital']
        if not recital:
            continue
        if recital == "undefined" :
            continue
        recitals.append(recital)
        if article  != "undefined" :
            article_dic[article].append(count)
        texts.append(recital)
        #glove_embeddings.append(tokens_to_embeddings(filter_tokens(recital), model))
        count += 1
    bert_embeddings = get_bert_embeddings(recitals, model)
    #embeddings.append(np.mean(sentence_embeddings, 0))
    return article_dic, texts, glove_embeddings, bert_embeddings

chapter_gdpr, section_gdpr, article_gdpr = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)

stop_words = stopwords.words('english')
tknzr = TweetTokenizer()
lemmatizer = WordNetLemmatizer()


embedding_method = EmbeddingMethod.bert
if embedding_method == EmbeddingMethod.glove:
    model = load_glove_vectors()
elif embedding_method == EmbeddingMethod.fasttext:
    model = load_fasttext_vectors()
else:
    model = load_bert_model("/home/yaguang/pretrained_models/uncased_L-12_H-768_A-12")
    model.max_seq_length = 512

#idpr_article_dic, idpr_text, idpr_glove_embeddings, idpr_bert_embeddings = get_embeddings(model, "data/GDPR-EN-Indian-converted.csv")
bdpr_article_dic, bdpr_text, bdpr_glove_embeddings, bdpr_bert_embeddings = get_embeddings(model, "data/LGPD-ES-Brazil-converted.csv")
gdpr_article_dic, gdpr_text, gdpr_glove_embeddings, gdpr_bert_embeddings = get_embeddings(model, "data/GDPR-EN-Europe-converted.csv")
w = open("simi_sentence.csv", 'w')
w.write("\t".join(["gdpr recital", "bdpr recital", "similarity"])+"\n")
for i in range(len(bdpr_bert_embeddings)):
    for j in range(len(gdpr_bert_embeddings)):
        sim = 1 - spatial.distance.cosine(bdpr_bert_embeddings[i], gdpr_bert_embeddings[j])
        if sim > 0.8:
            w.write("\t".join([bdpr_text[i], gdpr_text[j], str(sim)])+"\n")
w.close()
w = open("simi_article.csv", 'w')
w.write("\t".join(["gdpr article", "bdpr article", "similarity"])+"\n")
for key2 in gdpr_article_dic:
    for key1 in bdpr_article_dic:
        b_idx = bdpr_article_dic[key1]
        b_texts = ".".join([key1]+[bdpr_text[idx] for idx in b_idx])
        b_sum = np.sum([bdpr_bert_embeddings[idx] for idx in b_idx], 0)

        g_idx = gdpr_article_dic[key2]
        g_texts = ".".join([key2]+[gdpr_text[idx] for idx in g_idx])
        g_sum = np.sum([gdpr_bert_embeddings[idx] for idx in g_idx], 0)

        sim = 1 - spatial.distance.cosine(b_sum, g_sum)
        #if sim > 0.8:
        w.write("\t".join([b_texts, g_texts, str(sim)])+"\n")
w.close()
