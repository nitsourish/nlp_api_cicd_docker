from flask import Flask, render_template, flash, request
import requests
import json
import tensorflow as tf

app = Flask(__name__)

url = 'http://localhost:8501/v1/models/123:predict'
cat_map_rev = {0: 'ham', 1: 'spam'}
max_len = 150
def one_hot(text):
    return tf.keras.preprocessing.text.one_hot(input_text=text,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ',n=8000)

def make_prediction(string):
   encoded_s = one_hot(string)
   instances = tf.keras.preprocessing.sequence.pad_sequences([encoded_s],maxlen=max_len)
   data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
   headers = {"content-type": "application/json"}
   json_response = requests.post(url, data=data, headers=headers)
   predictions = json.loads(json_response.text)['predictions'][0][0]
   predictions = cat_map_rev[int(predictions>=0.5)]
   return predictions

@app.route("/predict",methods=["GET","POST"]) #1
def predict():
    instance = request.json
    instance = list(instance.values())[0]
    # print(instance)
    response = make_prediction(instance)
    return {'label': response}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
  