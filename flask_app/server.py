from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

# Enabling memory growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Importing NLP preprocessor
from src.preprocessing.nlp_preprocessing import NLP_Preprocessor

nlp_preprocessor = NLP_Preprocessor()
# Loading tokenizer
with open('models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)


def get_model():
    global model
    print('Loading the model')
    model = load_model('models/model.h5')

    print('The model is loaded')


def preprocess_headline(headline):
    text_data = nlp_preprocessor.split_sent([headline])
    tokens = nlp_preprocessor.tokenize(text_data)
    tokens = nlp_preprocessor.lowercase(tokens)
    tokens = nlp_preprocessor.lemmatize_tokens(tokens)
    tokens = nlp_preprocessor.remove_stopwords(tokens)
    tokens = nlp_preprocessor.remove_punctuation(tokens)
    text_data = nlp_preprocessor.tokens_to_text(tokens)

    headline_preproc = text_data

    headline_preproc = tokenizer.texts_to_sequences(headline_preproc)

    MAXLEN = 12
    headline_preproc = pad_sequences(headline_preproc, maxlen=MAXLEN, padding='post')

    # Sentiment analysis
    sentiment = nlp_preprocessor.get_sentiments([headline])

    return headline_preproc, sentiment


get_model()

headline_preproc, sentiment = preprocess_headline('Warmup')
model.predict([headline_preproc, sentiment])


@app.route('/')
def root_get():
    return render_template('score_headline.html')


@app.route('/', methods=['POST'])
def root_post():
    try:
        headline = request.get_json(force=True)['headline']
        score_to_word = lambda x: 'few' if x <= 0.5 else 'many'
        sentiment_to_word = lambda x: 'negative' if x <= 0 else 'positive'
        headline_preproc, sentiment = preprocess_headline(headline)
        views_pred, likes_pred, dislikes_pred = np.array(model.predict([headline_preproc, sentiment])).flatten()

        response = {
            'headline_score': f'A video with such a headline is predicted to have {score_to_word(views_pred)} views, {score_to_word(likes_pred)} likes,'
                              f' and {score_to_word(dislikes_pred)} dislikes. Sentiment of the headline is {sentiment_to_word(sentiment)}.\n'
        }
        return jsonify(response)
    except:
        pass


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="127.0.0.1", port=5000)
