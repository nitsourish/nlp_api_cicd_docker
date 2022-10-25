from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import string
import os

class data_process(BaseEstimator,TransformerMixin):
    def fit(self, data= None, y= None):
        return self
  
    def transform(self, data= None, y = None):
        data = data.strip()
        translate_table = dict((ord(char), None) for char in string.punctuation)   
        data = data.translate(translate_table)
        data = [' '.join(data.split(' '))]
        return data

class feature_extraction(BaseEstimator,TransformerMixin):
    def __init__(self, tf_idf_path = None):
        self.tf_idf_path = tf_idf_path
        
    def fit(self,data,y=None,test_data=None):
        return self

    def transform(self, data= None, y = None):
        tf = pickle.load(open(self.tf_idf_path,"rb"))
        data = tf.transform(data)
        return data

class prediction(BaseEstimator,TransformerMixin):
    def __init__(self,model_path=None,cat_map_rev = {0: 'ham', 1: 'spam'}):
        self.model_path = model_path
        self.cat_map_rev = cat_map_rev
    def fit(self,data,y=None):
        return self 
        
    def transform(self,data=None,y=None):
        model = pickle.load(open(os.path.join(self.model_path, 'clf.pkl'),"rb"))
        return self.cat_map_rev[model.predict(data)[0]]