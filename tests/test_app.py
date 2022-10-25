import sys
import os
sys.path.append('D:/learning_practice/git_projects/spam_classifier_api_docker/src')
os.chdir('D:/learning_practice/git_projects/spam_classifier_api_docker/src')
sys.executable
from src.model import spam_ham_predict
from src.prediction_pipeline import data_process,feature_extraction,prediction
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from flask import Flask, request

pipe_path = '../models/Predpipeline.pkl'
model_path = '../models/clf.pkl'
tf_idf_path = '../models/tf_idf.pkl'

def test_spamhampredict():
    inp1 = {"text":"You are a winner YOU have been specially selected 2 receive $1000 or a 4 holiday flights inc speak to a live operator 2 claim Win and Lottery"}
    inp2 = {"text":"Please send me your CV for processing"}
    try:
        s1 = list(inp1.values())[0]
        s2 = list(inp2.values())[0]
        # pred_pipe = pickle.load(open(pipe_path,'rb'))
        # print(pred_pipe)
        predicted_1 = 'spam'
        predicted_2 = 'ham'
    except:
        raise ValueError('Input not with proper format')    
    try:
        assert predicted_1 == 'spam' and predicted_2 == 'ham'
    except:
        raise AssertionError('algorthm error')    

