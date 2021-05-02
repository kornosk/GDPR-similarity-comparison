cosine_function = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

train_set = ["The sky is blue.", "The sun is bright."] #Documents
test_set = ["The abc sun in the sky is bright."] #Query
stopWords = stopwords.words('english')

vectorizer = CountVectorizer(stop_words = stopWords)
#print vectorizer
transformer = TfidfTransformer()
#print transformer

trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
testVectorizerArray = vectorizer.transform(test_set).toarray()
#print 'Fit Vectorizer to train set', trainVectorizerArray
#print 'Transform Vectorizer to test set', testVectorizerArray
cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

for vector in trainVectorizerArray:
    print (vector)
    for testV in testVectorizerArray:
        print (testV)
        cosine = cx(vector, testV)
        print (cosine)
transformer.fit(trainVectorizerArray)
"""
print
print (transformer.transform(trainVectorizerArray).toarray())

transformer.fit(testVectorizerArray)
print 
tfidf = transformer.transform(testVectorizerArray)
print (tfidf.todense())
"""
