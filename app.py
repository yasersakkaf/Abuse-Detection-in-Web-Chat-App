# app.py
from flask import Flask, request, jsonify
import joblib

model = joblib.load('profanity-check/profanity_check/data/model.joblib')
vectorizer = joblib.load('profanity-check/profanity_check/data/vectorizer.joblib')
 
 
app = Flask(__name__)


@app.route('/', methods=['POST'])
def infer():
    text_data = request.form.get('text')
    data = [text_data]
    vect = vectorizer.transform(data)
    p = model.predict(vect)[0]
    response = dict()
    response['result'] = 'Offensive' if p == 1 else 'Not Offensive'
    return jsonify(response)


if __name__ == "__main__":
    app.run('0.0.0.0', 8081)
