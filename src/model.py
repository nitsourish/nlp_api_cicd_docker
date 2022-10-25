import sys
sys.executable
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from json import JSONDecodeError, JSONEncoder
import pandas as pd
import pickle
import os
import collections
from sklearn import metrics
import sklearn
import numpy as np


class spam_ham_predict:
    def __init__(self, clf_path,tf_idf_path):
        self.clf = pickle.load(open(clf_path,'rb'))
        self.tf_idf = pickle.load(open(tf_idf_path,'rb'))
    def Featurize(self,text):
        features = self.tf_idf.transform(text)
        return features
    def Predict(self,features):    
        return self.clf.predict(features)[0]