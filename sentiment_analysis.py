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
        # regex expressions
        self.REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

        # files for input data (pre-split into training, development and testing)
        f_train_pos = open('IMDb/train/imdb_train_pos.txt','r', encoding="UTF-8")
        f_train_neg = open('IMDb/train/imdb_train_neg.txt','r', encoding="UTF-8")

        f_test_pos  = open('IMDb/test/imdb_test_pos.txt','r', encoding="UTF-8")
        f_test_neg  = open('IMDb/test/imdb_test_neg.txt','r', encoding="UTF-8")

        f_dev_pos   = open('IMDb/dev/imdb_dev_pos.txt','r', encoding="UTF-8")
        f_dev_neg   = open('IMDb/dev/imdb_dev_neg.txt','r', encoding="UTF-8")

        self.train_pos = []
        self.train_neg = []
        self.test_pos  = []
        self.test_neg  = []
        self.dev_pos   = []
        self.dev_neg   = []

        # read in per line
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
            
        # clean up reviews using regex expressions
        self.train_pos = self.preprocess_reviews(self.train_pos)
        self.train_neg = self.preprocess_reviews(self.train_neg)
        self.test_pos  = self.preprocess_reviews(self.test_pos)
        self.test_neg  = self.preprocess_reviews(self.test_neg)
        self.dev_pos   = self.preprocess_reviews(self.dev_pos)
        self.dev_neg   = self.preprocess_reviews(self.dev_neg)

        self.train_set = []
        self.test_set  = []
        self.dev_set   = []

        # create sets for data zipped with label (1 for positive, 0 for negative)
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
        # have ground truth in separate sets for easy passing to training, predictions etc.  
        self.Y_train = [x[1] for x in self.train_set]
        self.Y_test = [x[1] for x in self.test_set]
        self.Y_dev = [x[1] for x in self.dev_set]

    # DEFINE GLOBAL FUNCTIONS

    # lemmatise a string
    def get_list_tokens(self, string):
        sentence_split=nltk.tokenize.sent_tokenize(string) # split into sentences
        list_tokens=[]
        for sentence in sentence_split:
            list_tokens_sentence=nltk.tokenize.word_tokenize(sentence) # split into words - list of list of words where each list is a sentence
            for token in list_tokens_sentence:
                list_tokens.append(self.lemmatizer.lemmatize(token).lower()) # lemmatize and lowercase each token
                
        return list_tokens

    # create a lemmatizer
    def get_lemmatizer(self):
        return nltk.stem.WordNetLemmatizer()

    # create stopwords
    def get_stopwords(self):
        stopwords=set(nltk.corpus.stopwords.words('english'))
        return stopwords
        
    # create word-frequency dictionary (vocabulary)
    def get_vocabulary(self, training_set, num_features):
        dict_word_frequency={}
        for instance in training_set:
            sentence_tokens=self.get_list_tokens(instance[0]) # get the tokenized, lemmatized review
            for word in sentence_tokens:
                if word in self.stopwords: continue # if it's a stopword, don't add
                if word not in dict_word_frequency: dict_word_frequency[word]=1 # if not an entry, make an entry with value 1, else add 1 to current frequency
                else: dict_word_frequency[word]+=1
        sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features] # sort by frequency, keep top num_features words
        vocabulary=[]
        for word,frequency in sorted_list:
            vocabulary.append(word) # create vocabulary list from top words in dictionary
        return vocabulary

    # get vector representation of a review using a create vocabulary
    def get_vector_text_all(self, list_vocab, string):
        vector_text=np.zeros(len(list_vocab)+4)
        list_tokens_string=self.get_list_tokens(string) # tokenize and lemmatize review
        for i, word in enumerate(list_vocab):
            if word in list_tokens_string:
                vector_text[i]=list_tokens_string.count(word) # if word in vocab is in review, set the element to the number of times that word appears in current review


        p_scores = self.vader.polarity_scores(self.list_to_sentence(list_tokens_string)) # VADER sentiment-analysis scores
        vector_text[i+1] = p_scores['neg']
        vector_text[i+2] = p_scores['neu']
        vector_text[i+3] = p_scores['pos']
        vector_text[i+4] = p_scores['compound']
        return vector_text

    # given a list of strings, create a sentence 
    def list_to_sentence(self, list_string):
        str_rtn = ""
        for word in list_string:
            str_rtn += word + " "
        return str_rtn

    # write elements of list to lines of a file
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
        self.tfidf_vec = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True, max_features=4000)
        self.tfX = self.tfidf_vec.fit_transform(self.train_pos+self.train_neg)
        self.tfX_test = self.tfidf_vec.transform(self.test_pos+self.test_neg)
        self.tfX_dev = self.tfidf_vec.transform(self.dev_pos+self.dev_neg)

        # combining features
        self.tfX_reshape = scipy.sparse.csr_matrix.toarray(self.tfX)
        self.tfX_test_reshape = scipy.sparse.csr_matrix.toarray(self.tfX_test)
        self.tfX_dev_reshape = scipy.sparse.csr_matrix.toarray(self.tfX_dev)

        # create copies 
        self.Xvec_all = self.Xvec.copy()
        self.Xvec_all_std = self.Xvec.copy()
        self.Xvec_all_test = self.Xvec_test.copy()
        self.Xvec_all_test_std = self.Xvec_test.copy()
        self.Xvec_all_dev = self.Xvec_dev.copy()
        self.Xvec_all_dev_std = self.Xvec_dev.copy()

        # append TF-IDF features to the vector representation and VADER scores
        for i in range(0, len(self.tfX_reshape)):
            self.Xvec_all[i] = (np.append(self.Xvec_all[i][0], np.asarray(self.tfX_reshape[i])), self.Xvec_all[i][1])
        for i in range(0, len(self.tfX_test_reshape)):
            self.Xvec_all_test[i] = np.append(self.Xvec_all_test[i][0], np.asarray(self.tfX_test_reshape[i]))
        for i in range(0, len(self.tfX_dev_reshape)):
            self.Xvec_all_dev[i] = np.append(self.Xvec_all_dev[i][0], np.asarray(self.tfX_dev_reshape[i]))
            
        # create a standardizer fitted to the training data, and transform the training, test and development set to be standardized
        self.scaler = sklearn.preprocessing.StandardScaler()    
        self.nx_all = [x[0] for x in self.Xvec_all]
        self.std_x = self.scaler.fit_transform(self.nx_all)
        self.std_x_test = self.scaler.transform(self.Xvec_all_test)
        self.std_x_dev = self.scaler.transform(self.Xvec_all_dev)

        # create a PCA transformer with 50 components fitted to the training data, and transform the training, test and development set to be projected to the lower-dimension
        self.pca_transformer = sklearn.decomposition.PCA(n_components=50)
        self.pca_x = self.pca_transformer.fit_transform(self.std_x)
        self.pca_x_test = self.pca_transformer.transform(self.std_x_test)
        self.pca_x_dev = self.pca_transformer.transform(self.std_x_dev)

    def train_svms(self):
        # TRAIN SVMS

        # for without PCA
        self.svm_clf = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=0.8)
        t = time.perf_counter()
        self.svm_clf.fit(self.std_x, self.Y_train)
        self.t_std = time.perf_counter() - t

        # for with pca
        self.svm_clf_pca = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=0.8)
        t = time.perf_counter()
        self.svm_clf_pca.fit(self.pca_x, self.Y_train)
        self.t_pca = time.perf_counter() - t      

    def make_predictions(self):
        # MAKE PREDICTIONS

        # without PCA
        t = time.perf_counter()
        self.preds = self.svm_clf.predict(self.std_x_test)
        self.t_std_pred = time.perf_counter() - t

        # with PCA
        t = time.perf_counter()
        self.preds_pca = self.svm_clf_pca.predict(self.pca_x_test)
        self.t_pca_pred = time.perf_counter() - t    

    def show_metrics(self):
        # SHOW METRICS 

        print(sklearn.metrics.classification_report(self.Y_test, self.preds))
        print()
        print(sklearn.metrics.classification_report(self.Y_test, self.preds_pca))
        print()
        # timings for training and predicting for with and without PCA
        print("Without PCA learn t=",self.t_std,"   predict t=",self.t_std_pred)
        print("With PCA learn t=",self.t_pca,"   predict t=",self.t_pca_pred)

    # save results of both SVMs trained in train_svms()
    def save_training_results(self):

        with open('training_results.txt','w') as f:
            f.write("Metrics without PCA")
            f.write("\n")
            f.write(sklearn.metrics.classification_report(self.Y_test, self.preds))
            f.write("\n")
            f.write("Metrics with PCA, n_components=50")
            f.write("\n")
            f.write(sklearn.metrics.classification_report(self.Y_test, self.preds_pca))
            f.write("\n")
            f.write("Without PCA learn t=" + str(self.t_std) + "   predict t=" + str(self.t_std_pred))
            f.write("\n")
            f.write("With PCA learn t=" + str(self.t_pca) + "   predict t=" + str(self.t_pca_pred))

    # save models to files (serialized objects) which can be opened in another program
    def save_models(self):
        with open('models/vocab_model.txt', 'wb') as f:
            pickle.dump(self.vocabulary, f)
        with open('models/scaler_model.txt', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('models/pca_transformer_model.txt', 'wb') as f:
            pickle.dump(self.pca_transformer, f)
        with open('models/svm_std.txt', 'wb') as f:
            pickle.dump(self.svm_clf, f)
        with open('models/svm_std_pca.txt', 'wb') as f:
            pickle.dump(self.svm_clf_pca, f)


    # DEV SET PARAMETER OPTIMISATION - simple loop through and timing/accuracy scoring for various hyperparameters


    def optimise_pca(self):
        # PCA COMPONENTS

        self.comp_list = [5,10,20,50,100,500,1000]
        self.acc_list_comp = [] # accuracy
        self.t_learn_list_comp = [] # time to train svm
        self.pred_list_comp = [] # time to predict svm

        for n in self.comp_list:

            pca_transformer = sklearn.decomposition.PCA(n_components=n)
            pca_x = pca_transformer.fit_transform(self.std_x)
            pca_x_dev = pca_transformer.transform(self.std_x_dev)

            t = time.perf_counter()
            svm_clf_pca = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=0.8)
            svm_clf_pca.fit(pca_x, self.Y_train)
            t_pca = time.perf_counter() - t

            t = time.perf_counter()
            preds_pca = svm_clf_pca.predict(pca_x_dev)
            t_pca_pred = time.perf_counter() - t
            
            self.acc_list_comp.append(sklearn.metrics.accuracy_score(self.Y_dev, preds_pca))
            self.t_learn_list_comp.append(t_pca)
            self.pred_list_comp.append(t_pca_pred)
            
            print(n)            

    def optimise_vocab_feature(self):
        # VOCAB FEATURE SIZE

        self.vocab_list = [100,200,500,1000,1500,2000,3000,4000]
        self.acc_list_voc = [] # accuracy
        self.t_vectorize_list_voc = [] # time to pre-process
        self.t_learn_list_voc = [] # time to train svm
        self.t_transform_voc = [] # time to project data to lower dimension with PCA

        for vf in self.vocab_list:
            t = time.perf_counter()
            vocabulary = self.get_vocabulary(self.train_set, vf)
            # vector count and VADER analysis
            Xvec = [(self.get_vector_text_all(vocabulary, x[0]), x[1]) for x in self.train_set]
            Xvec_dev = [(self.get_vector_text_all(vocabulary, x[0]), x[1]) for x in self.dev_set]

            # TF-IDF vectorisation
            tfidf_vec = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True, max_features=2000)
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
            

            pca_transformer = sklearn.decomposition.PCA(n_components=50)
            pca_x = pca_transformer.fit_transform(std_x)

            t2 = time.perf_counter()
            std_x_dev = scaler.transform(Xvec_all_dev)
            pca_x_dev = pca_transformer.transform(std_x_dev)
            t_trans = time.perf_counter() - t2

            t_vec = time.perf_counter() - t
            
            t = time.perf_counter()
            svm_clf = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=0.8)
            svm_clf.fit(pca_x, self.Y_train)
            t_learn = time.perf_counter() - t

            preds = svm_clf.predict(pca_x_dev)
            
            
            self.acc_list_voc.append(sklearn.metrics.accuracy_score(self.Y_dev, preds))
            self.t_learn_list_voc.append(t_learn)
            self.t_vectorize_list_voc.append(t_vec)
            self.t_transform_voc.append(t_trans)
            
            print(vf)  

    def optimise_tfidf(self):
        self.tfidf_list = [100,200,500,1000,1500,2000,3000,4000]
        self.acc_list_tf = [] # accuracy
        self.t_vectorize_list_tf = [] # time to pre-process
        self.t_learn_list_tf = [] # time to train svm
        self.t_transform_tf = [] # time to project data to lower dimension with PCA

        for tf in self.tfidf_list:
            t = time.perf_counter()
            vocabulary = self.get_vocabulary(self.train_set, 2000)
            # vector count and VADER analysis
            Xvec = [(self.get_vector_text_all(vocabulary, x[0]), x[1]) for x in self.train_set]
            Xvec_dev = [(self.get_vector_text_all(vocabulary, x[0]), x[1]) for x in self.dev_set]

            # TF-IDF vectorisation
            tfidf_vec = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True, max_features=tf)
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
            
            pca_transformer = sklearn.decomposition.PCA(n_components=50)
            pca_x = pca_transformer.fit_transform(std_x)

            t2 = time.perf_counter()
            std_x_dev = scaler.transform(Xvec_all_dev)
            pca_x_dev = pca_transformer.transform(std_x_dev)
            t_trans = time.perf_counter() - t2
            
            t_vec = time.perf_counter() - t
            
            t = time.perf_counter()
            svm_clf = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=0.8)
            svm_clf.fit(pca_x, self.Y_train)
            t_learn = time.perf_counter() - t

            preds = svm_clf.predict(pca_x_dev)
            
            self.acc_list_tf.append(sklearn.metrics.accuracy_score(self.Y_dev, preds))
            self.t_learn_list_tf.append(t_learn)
            self.t_vectorize_list_tf.append(t_vec)
            self.t_transform_tf.append(t_trans)
            
            print(tf)

    def optimise_c(self):
        # SVM C REGULARISATION PARAMETER 

        self.c_list = [2.0,1.5,1.0,0.9,0.8,0.7,0.6,0.4,0.2,0.1]
        self.acc_list_c = [] # accuracy
        self.t_learn_list_c = [] # time to train svm

        pca_transformer = sklearn.decomposition.PCA(n_components=50)
        pca_x = pca_transformer.fit_transform(self.std_x)
        pca_x_dev = pca_transformer.transform(self.std_x_dev)

        for c in self.c_list:
            t = time.perf_counter()
            svm_clf_pca = sklearn.svm.SVC(kernel='rbf', gamma='scale', C=c)
            
            svm_clf_pca.fit(pca_x, self.Y_train)
            t_pca = time.perf_counter() - t
        
            t = time.perf_counter()
            preds_pca = svm_clf_pca.predict(pca_x_dev)
            t_pca_pred = time.perf_counter() - t
            
            self.acc_list_c.append(sklearn.metrics.accuracy_score(self.Y_dev, preds_pca))
            self.t_learn_list_c.append(t_pca)
            
            print(c)  

    def optimise_gamma(self):
        self.gamma_list = ['scale',0.0001,0.001,0.01,0.05,0.08,0.1,0.2,0.3]
        self.acc_list_gamma = [] # accuracy
        self.t_learn_list_gamma = [] # time to train svm

        pca_transformer = sklearn.decomposition.PCA(n_components=50)
        pca_x = pca_transformer.fit_transform(self.std_x)
        pca_x_dev = pca_transformer.transform(self.std_x_dev)

        for g in self.gamma_list:
            t = time.perf_counter()
            svm_clf_pca = sklearn.svm.SVC(kernel='rbf', gamma=g, C=0.8)
            
            svm_clf_pca.fit(pca_x, self.Y_train)
            t_pca = time.perf_counter() - t
        
            t = time.perf_counter()
            preds_pca = svm_clf_pca.predict(pca_x_dev)
            t_pca_pred = time.perf_counter() - t
            
            self.acc_list_gamma.append(sklearn.metrics.accuracy_score(self.Y_dev, preds_pca))
            self.t_learn_list_gamma.append(t_pca)
            
            print(g)  

    def save_optimisation_results(self):
        # SAVE RESULTS 
            
        self.write_file(self.comp_list, 'Dev_results/comp_list.txt')
        self.write_file(self.c_list, 'Dev_results/c_list.txt')
        self.write_file(self.vocab_list, 'Dev_results/vocab_list.txt')
        self.write_file(self.tfidf_list, 'Dev_results/tfidf_list.txt')
        self.write_file(self.gamma_list, 'Dev_results/gamma_list.txt')

        self.write_file(self.acc_list_comp, 'Dev_results/acc_list_comp.txt')
        self.write_file(self.acc_list_c, 'Dev_results/acc_list_c.txt')
        self.write_file(self.acc_list_voc, 'Dev_results/acc_list_voc.txt')
        self.write_file(self.acc_list_tf, 'Dev_results/acc_list_tf.txt')
        self.write_file(self.acc_list_gamma, 'Dev_results/acc_list_gamma.txt')


        self.write_file(self.t_learn_list_comp, 'Dev_results/t_learn_list_comp.txt')
        self.write_file(self.t_learn_list_c, 'Dev_results/t_learn_list_c.txt')
        self.write_file(self.t_learn_list_voc, 'Dev_results/t_learn_list_voc.txt')
        self.write_file(self.t_vectorize_list_voc, 'Dev_results/t_vectorize_list_voc.txt')     
        self.write_file(self.t_learn_list_tf, 'Dev_results/t_learn_list_tf.txt')
        self.write_file(self.t_vectorize_list_tf, 'Dev_results/t_vectorize_list_tf.txt')
        self.write_file(self.t_learn_list_gamma, 'Dev_results/t_learn_list_gamma.txt')

        self.write_file(self.t_transform_voc, 'Dev_results/t_transform_voc.txt')
        self.write_file(self.t_transform_tf, 'Dev_results/t_transform_tf')


# METHODS TO CALL IN ORDER
# vectorize_input_data()
# train_svms()
# make_predictions()
# show_metrics()
#
# DEVELOPMENT SET OPTIMISATION METHODS
# optimise_pca()
# optimise_c()
# optimise_vocab_feature()
# optimise_tfidf()
# save_optimisation_results()

if __name__ == '__main__':
    s = SentimentAnalysis()
    # OR LOAD A SERIALISED OBJECT
    print("Started")
    #s = []
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
    s.save_training_results()
    print("Results saved")
    s.save_models()
    print("Models saved")

    print("Development set optimisation started")
    s.optimise_pca()
    print("PCA components optimised")
    s.optimise_c()
    print("C regularisation optimised")
    s.optimise_vocab_feature()
    print("Vocabulary feature size optimised")
    s.optimise_tfidf()
    print("TF-IDF feature size optimised")
    s.optimise_gamma()
    print("Gamma optimised")
    s.save_optimisation_results()
    print("Optimisation results saved")

    # save object
    with open('sentiment_object.txt', 'wb') as f:
        pickle.dump(s, f)

    print("SentimentAnalysis object saved")



               











 







