
import sys
sys.executable
import pickle
from flask import Flask, request
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from model import spam_ham_predict
from prediction_pipeline import data_process,feature_extraction,prediction

app = Flask(__name__)
# model_path = 'models/clf.pkl'
pipe_path = 'models/Predpipeline.pkl'
model_path = 'models/'
tf_idf_path = 'models/tf_idf.pkl'
# sh = spam_ham_predict(clf_path=model_path,tf_idf_path=tf_idf_path)

@app.route("/predict", methods=['POST'])
def predict():

    labels = ['ham', 'spam']

    instance = request.json
    
    pred_pipe = pickle.load(open(pipe_path,'rb'))
    predicted_label = str(pred_pipe.transform(list(instance.values())[0]))

    # features =  sh.Featurize(list(instance.values()))

    # predicted_index = sh.Predict(features)

    # predicted_label = labels[predicted_index]

    return {'label': predicted_label}


if __name__ == "__main__":
    app.run(port=5000, debug=True,host='0.0.0.0')
    