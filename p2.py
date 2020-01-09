import matplotlib.pyplot as plt
import numpy as np
import nltk
import sklearn
from sklearn import decomposition, feature_extraction, preprocessing, svm, metrics
import scipy
import pandas as pd
import math
import operator
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import time
import sys

class SentimentAnalysis():

    # IMPORT DATA

    def preprocess_reviews(self, reviews):
        reviews = [self.REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [self.REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
            
        return reviews

    def __init__(self):
        self.REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

        
        f_train_pos = open('IMDb/train/imdb_train_pos.txt','r', encoding="UTF-8")
        f_train_neg = open('IMDb/train/imdb_train_neg.txt','r', encoding="UTF-8")

        f_test_pos  = open('IMDb/test/imdb_test_pos.txt','r', encoding="UTF-8")
        f_test_neg  = open('IMDb/test/imdb_test_neg.txt','r', encoding="UTF-8")

        f_dev_pos   = open('IMDb/dev/imdb_dev_pos.txt','r', encoding="UTF-8")
        f_dev_neg   = open('IMDb/dev/imdb_dev_neg.txt','r', encoding="UTF-8")

        p_in = open('positive.txt','r')
        n_in = open('negative.txt','r')

        self.train_pos = []
        self.train_neg = []
        self.test_pos  = []
        self.test_neg  = []
        self.dev_pos   = []
        self.dev_neg   = []

        for line in f_train_pos:
            self.train_pos.append(line)
        for line in f_train_neg:
            self.train_neg.append(line)
        for line in f_test_pos:
            self.test_pos.append(line)
        for line in f_test_neg:
            self.test_neg.append(line)
        for line in f_dev_pos:
            self.dev_pos.append(line)
        for line in f_dev_neg:
            self.dev_neg.append(line)
            

        self.train_pos = self.preprocess_reviews(self.train_pos)
        self.train_neg = self.preprocess_reviews(self.train_neg)
        self.test_pos  = self.preprocess_reviews(self.test_pos)
        self.test_neg  = self.preprocess_reviews(self.test_neg)
        self.dev_pos   = self.preprocess_reviews(self.dev_pos)
        self.dev_neg   = self.preprocess_reviews(self.dev_neg)

        self.train_set = []
        self.test_set  = []
        self.dev_set   = []

        self.train_set += [(x,1) for x in self.train_pos]
        self.train_set += [(x,0) for x in self.train_neg]
        self.test_set  += [(x,1) for x in self.test_pos]
        self.test_set  += [(x,0) for x in self.test_neg]
        self.dev_set   += [(x,1) for x in self.dev_pos]
        self.dev_set   += [(x,0) for x in self.dev_neg]

        # DEFINE GLOBAL VARIABLES

        self.lemmatizer = self.get_lemmatizer()
        self.stopwords = self.get_stopwords()
        self.vocabulary = self.get_vocabulary(self.train_set, 2000)
        self.vader = SentimentIntensityAnalyzer()
        self.Y_train = [x[1] for x in self.train_set]
        self.Y_test = [x[1] for x in self.test_set]
        self.Y_dev = [x[1] for x in self.dev_set]

    # DEFINE GLOBAL FUNCTIONS

    def get_list_tokens(self, string):
        sentence_split=nltk.tokenize.sent_tokenize(string)
        list_tokens=[]
        for sentence in sentence_split:
            list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
            for token in list_tokens_sentence:
                list_tokens.append(self.lemmatizer.lemmatize(token).lower())
                
        return list_tokens

    def get_lemmatizer(self):
        return nltk.stem.WordNetLemmatizer()

    def get_stopwords(self):
        stopwords=set(nltk.corpus.stopwords.words('english'))
        return stopwords
        
    def get_vocabulary(self, training_set, num_features): # Function to retrieve vocabulary
        dict_word_frequency={}
        for instance in training_set:
            sentence_tokens=self.get_list_tokens(instance[0])
            for word in sentence_tokens:
                if word in self.stopwords: continue
                if word not in dict_word_frequency: dict_word_frequency[word]=1
                else: dict_word_frequency[word]+=1
        sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
        vocabulary=[]
        for word,frequency in sorted_list:
            vocabulary.append(word)
        return vocabulary

    def get_vector_text_all(self, list_vocab, string):
        vector_text=np.zeros(len(list_vocab)+4)
        list_tokens_string=self.get_list_tokens(string)
        for i, word in enumerate(list_vocab):
            if word in list_tokens_string:
                vector_text[i]=list_tokens_string.count(word)
        p_scores = self.vader.polarity_scores(self.list_to_sentance(list_tokens_string))
        vector_text[i+1] = p_scores['neg']
        vector_text[i+2] = p_scores['neu']
        vector_text[i+3] = p_scores['pos']
        vector_text[i+4] = p_scores['compound']
        return vector_text

    def list_to_sentance(self, list_string):
        str_rtn = ""
        for word in list_string:
            str_rtn += word + " "
        return str_rtn

    def write_file(self, l, str_file):
        f = open(str_file, 'w')
        f.writelines([str(i) + "\n" for i in l])



    # VECTORIZE INPUT DATA
    def vectorize_input_data(self):
        # vector count and VADER analysis
        self.Xvec = [(self.get_vector_text_all(self.vocabulary, x[0]), x[1]) for x in self.train_set]
        self.Xvec_test = [(self.get_vector_text_all(self.vocabulary, x[0]), x[1]) for x in self.test_set]
        self.Xvec_dev = [(self.get_vector_text_all(self.vocabulary, x[0]), x[1]) for x in self.dev_set]

        # TF-IDF vectorisation
        self.tfidf_vec = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True, max_features=500)
        self.tfX = self.tfidf_vec.fit_transform(self.train_pos+self.train_neg)
        self.tfX_test = self.tfidf_vec.transform(self.test_pos+self.test_neg)
        self.tfX_dev = self.tfidf_vec.transform(self.dev_pos+self.dev_neg)

        # combining features
        self.tfX_reshape = scipy.sparse.csr_matrix.toarray(self.tfX)
        self.tfX_test_reshape = scipy.sparse.csr_matrix.toarray(self.tfX_test)
        self.tfX_dev_reshape = scipy.sparse.csr_matrix.toarray(self.tfX_dev)

        self.Xvec_all = self.Xvec.copy()
        self.Xvec_all_std = self.Xvec.copy()
        self.Xvec_all_test = self.Xvec_test.copy()
        self.Xvec_all_test_std = self.Xvec_test.copy()
        self.Xvec_all_dev = self.Xvec_dev.copy()
        self.Xvec_all_dev_std = self.Xvec_dev.copy()

        for i in range(0, len(self.tfX_reshape)):
            self.Xvec_all[i] = (np.append(self.Xvec_all[i][0], np.asarray(self.tfX_reshape[i])), self.Xvec_all[i][1])
        for i in range(0, len(self.tfX_test_reshape)):
            self.Xvec_all_test[i] = np.append(self.Xvec_all_test[i][0], np.asarray(self.tfX_test_reshape[i]))
        for i in range(0, len(self.tfX_dev_reshape)):
            self.Xvec_all_dev[i] = np.append(self.Xvec_all_dev[i][0], np.asarray(self.tfX_dev_reshape[i]))
            
        self.scaler = sklearn.preprocessing.StandardScaler()    
        self.nx_all = [x[0] for x in self.Xvec_all]
        self.std_x = self.scaler.fit_transform(self.nx_all)
        self.std_x_test = self.scaler.transform(self.Xvec_all_test)
        self.std_x_dev = self.scaler.transform(self.Xvec_all_dev)

        self.pca_transformer = sklearn.decomposition.PCA(n_components=20)
        self.pca_x = self.pca_transformer.fit_transform(self.std_x)
        self.pca_x_test = self.pca_transformer.transform(self.std_x_test)
        self.pca_x_dev = self.pca_transformer.transform(self.std_x_dev)

    def train_svms(self):
        # TRAIN SVMS

        self.svm_clf = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=0.8)
        t = time.perf_counter()
        self.svm_clf.fit(self.std_x, self.Y_train)
        self.t_std = time.perf_counter() - t

        self.svm_clf_pca = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=0.8)
        t = time.perf_counter()
        self.svm_clf_pca.fit(self.pca_x, self.Y_train)
        self.t_pca = time.perf_counter() - t      

    def make_predictions(self):
        # MAKE PREDICTIONS

        t = time.perf_counter()
        self.preds = self.svm_clf.predict(self.std_x_test)
        self.t_std_pred = time.perf_counter() - t

        t = time.perf_counter()
        self.preds_pca = self.svm_clf_pca.predict(self.pca_x_test)
        self.t_pca_pred = time.perf_counter() - t    

    def show_metrics(self):
        # SHOW METRICS 

        print(sklearn.metrics.classification_report(self.Y_test, self.preds))
        print()
        print(sklearn.metrics.classification_report(self.Y_test, self.preds_pca))
        print()
        print("Without PCA learn t=",self.t_std,"   predict t=",self.t_std_pred)
        print("With PCA learn t=",self.t_pca,"   predict t=",self.t_pca_pred)      


    # DEV SET PARAMETER OPTIMISATION


    def optimise_pca(self):
        # PCA COMPONENTS

        self.comp_list = [5,10,20,50,100,500,1000]
        self.acc_list_comp = []
        self.t_learn_list_comp = []
        self._pred_list_comp = []

        for n in self.comp_list:

            pca_transformer = sklearn.decomposition.PCA(n_components=n)
            pca_x = pca_transformer.fit_transform(self.std_x)
            #pca_x_test = pca_transformer.transform(std_x_test)
            pca_x_dev = pca_transformer.transform(self.std_x_dev)

            svm_clf_pca = sklearn.svm.SVC(kernel='rbf', gamma='scale')
            t_start_pca = time.perf_counter()
            svm_clf_pca.fit(pca_x, Y_train)
            t_pca = time.perf_counter() - t_start_pca

            t = time.perf_counter()
            preds_pca = svm_clf_pca.predict(pca_x_dev)
            t_pca_pred = time.perf_counter() - t
            
            self.acc_list_comp.append(sklearn.metrics.accuracy_score(self.Y_dev, preds_pca))
            self.t_learn_list_comp.append(t_pca)
            self.t_pred_list_comp.append(t_pca_pred)
            
            print(n)          

    def optimise_c(self):
        # SVM C REGULARISATION PARAMETER 

        self.c_list = [2.0,1.5,1.0,0.8,0.6,0.4,0.2,0.1]
        self.acc_list_c = []
        self.t_learn_list_c = []

        pca_transformer = sklearn.decomposition.PCA(n_components=20)
        pca_x = pca_transformer.fit_transform(self.std_x)
        pca_x_dev = pca_transformer.transform(self.std_x_dev)

        for c in c_list:
            svm_clf_pca = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=c)
            t_start_pca = time.perf_counter()
            svm_clf_pca.fit(pca_x, Y_train)
            t_pca = time.perf_counter() - t_start_pca

            t = time.perf_counter()
            preds_pca = svm_clf_pca.predict(pca_x_dev)
            t_pca_pred = time.perf_counter() - t
            
            self.acc_list_c.append(sklearn.metrics.accuracy_score(self.Y_dev, preds_pca))
            self.t_learn_list_c.append(t_pca)
            
            print(c)    

    def optimise_vocab_feature(self):
        # VOCAB FEATURE SIZE

        self.vocab_list = [100,200,500,1000,2000]
        self.acc_list_voc = []
        self.t_vectorize_list_voc = []
        self.t_learn_list_voc = []

        for vf in vocab_list:
            t = time.perf_counter()
            vocabulary = get_vocabulary(self.train_set, vf)
            # vector count and VADER analysis
            Xvec = [(get_vector_text_all(vocabulary, x[0]), x[1]) for x in self.train_set]
            Xvec_dev = [(get_vector_text_all(vocabulary, x[0]), x[1]) for x in self.dev_set]

            # TF-IDF vectorisation
            tfidf_vec = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True, max_features=500)
            tfX = tfidf_vec.fit_transform(self.train_pos+self.train_neg)
            tfX_dev = tfidf_vec.transform(self.dev_pos+self.dev_neg)

            # combining features
            tfX_reshape = scipy.sparse.csr_matrix.toarray(tfX)
            tfX_dev_reshape = scipy.sparse.csr_matrix.toarray(tfX_dev)

            Xvec_all = Xvec.copy()
            Xvec_all_std = Xvec.copy()
            Xvec_all_dev = Xvec_dev.copy()
            Xvec_all_dev_std = Xvec_dev.copy()

            for i in range(0, len(tfX_reshape)):
                Xvec_all[i] = (np.append(Xvec_all[i][0], np.asarray(tfX_reshape[i])), Xvec_all[i][1])
            for i in range(0, len(tfX_dev_reshape)):
                Xvec_all_dev[i] = np.append(Xvec_all_dev[i][0], np.asarray(tfX_dev_reshape[i]))
            
            scaler = sklearn.preprocessing.StandardScaler()    
            nx_all = [x[0] for x in Xvec_all]
            std_x = scaler.fit_transform(nx_all)
            std_x_dev = scaler.transform(Xvec_all_dev)

            pca_transformer = sklearn.decomposition.PCA(n_components=20)
            pca_x = pca_transformer.fit_transform(std_x)
            pca_x_dev = pca_transformer.transform(std_x_dev)
            
            t_vec = time.perf_counter() - t
            
            svm_clf_pca = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=0.8)
            t = time.perf_counter()
            svm_clf_pca.fit(pca_x, Y_train)
            t_pca = time.perf_counter() - t

            t = time.perf_counter()
            preds_pca = svm_clf_pca.predict(pca_x_dev)
            t_pca_pred = time.perf_counter() - t
            
            self.acc_list_voc.append(sklearn.metrics.accuracy_score(self.Y_dev, preds_pca))
            self.t_learn_list_voc.append(t_pca)
            self.t_vectorize_list_voc.append(t_vec)
            
            print(vf)      

    def save_optimisation_results(self):
        # SAVE RESULTS 
            
        self.write_file(self.comp_list, 'Dev_results/comp_list.txt')
        self.write_file(self.c_list, 'Dev_results/c_list.txt')
        self.write_file(self.vocab_list, 'Dev_results/vocab_list.txt')

        self.write_file(self.acc_list_comp, 'Dev_results/acc_list_comp.txt')
        self.write_file(self.acc_list_c, 'Dev_results/acc_list_c.txt')
        self.write_file(self.acc_list_voc, 'Dev_results/acc_list_voc.txt')

        self.write_file(self.t_learn_list_comp, 'Dev_results/t_learn_list_comp.txt')
        self.write_file(self.t_learn_list_c, 'Dev_results/t_learn_list_c.txt')
        self.write_file(self.t_learn_list_voc, 'Dev_results/t_learn_list_voc.txt')
        self.write_file(self.t_vectorize_list_voc, 'Dev_results/t_vectorize_list_voc.txt')     


# vectorize_input_data
# train_svms
# make_predictions
# show_metrics
#
#
# optimise_pca
# optimise_c
# optimise_vocab_feature
# save_optimisation_results


s = SentimentAnalysis()
#or
#with open('sentiment_object.txt', 'rb') as f:
#    s = pickle.load(f)

print("Init finished")
s.vectorize_input_data()
print("Data vectorised")
s.train_svms()
print("SVMs trained")
s.make_predictions()
print("Predictions made")
s.show_metrics()

# save object
with open('sentiment_object.txt', 'wb') as f:
    pickle.dump(s, f)

               











 







