import sys
import os

sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
import pickle
from src.preprocessing.nlp_preprocessing import NLP_Preprocessor

# Visualization libraries and configurations
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('axes', titlesize=18)  # fontsize of the axes title
plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
plt.rc('font', size=12)  # controls default text sizes
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.rc('legend', fontsize=12)  # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

# Enabling memory growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Seed
np.random.seed(4)
tf.random.set_seed(4)


# For testing user's headlines
def get_model():
    global model
    print('Loading models')
    model = load_model('models/model.h5')
    print('Models loaded')


# get_model()
# model.predict([np.array([list(range(12))]), np.array([1])])


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


nlp_preprocessor = NLP_Preprocessor()

# Loading the data
df = pd.read_csv('data/preprocessed/news.csv')

# Filter the data by date
filt = df['date'].between(7, 365)
df = df[filt]

X_headline = df['headline'].to_numpy(dtype=str)
X_headline_preproc = df['headline_preproc'].to_numpy(dtype=str)
X_sentiment = df['sentiment'].to_numpy()

# Turn X_headline to sequences of numbers
NUM_WORDS = 12000

tokenizer = Tokenizer(num_words=NUM_WORDS)

tokenizer.fit_on_texts(texts=X_headline_preproc)
X_headline_preproc = tokenizer.texts_to_sequences(X_headline_preproc)

# Make inputs the same length
MAXLEN = 12
X_headline_preproc = pad_sequences(X_headline_preproc, maxlen=MAXLEN, padding='post')


def get_y():
    global X_headline_preproc_train, X_headline_preproc_test, X_sentiment_train, X_sentiment_test, \
        y_views_train, y_views_test, y_likes_train, y_likes_test, y_dislikes_train, y_dislikes_test, \
        X_headline_preproc_val, X_sentiment_val, y_views_val, y_likes_val, y_dislikes_val

    y_views = df['views'].to_numpy()
    y_likes = df['likes'].to_numpy()
    y_dislikes = df['dislikes'].to_numpy()

    # Making labels binary
    threshold = np.median(y_views)
    y_views_bin = np.array([1 if yi > threshold else 0 for yi in y_views])
    threshold = np.median(y_likes)
    y_likes_bin = np.array([1 if yi > threshold else 0 for yi in y_likes])
    threshold = np.median(y_dislikes)
    y_dislikes_bin = np.array([1 if yi > threshold else 0 for yi in y_dislikes])

    # Train test split
    X_headline_preproc_train, X_headline_preproc_test, X_sentiment_train, X_sentiment_test, \
    y_views_train, y_views_test, y_likes_train, y_likes_test, y_dislikes_train, y_dislikes_test = \
        train_test_split(X_headline_preproc, X_sentiment, y_views_bin, y_likes_bin, y_dislikes_bin,
                         test_size=0.4, shuffle=True)

    X_headline_preproc_test, X_headline_preproc_val, X_sentiment_test, X_sentiment_val, \
    y_views_test, y_views_val, y_likes_test, y_likes_val, y_dislikes_test, y_dislikes_val = \
        train_test_split(X_headline_preproc_test, X_sentiment_test, y_views_test, y_likes_test, y_dislikes_test,
                         test_size=0.3, shuffle=True)


# Confusion matrix
def confusion_matrix(y_true, y_pred, model_type):
    t_pos = np.sum(np.bitwise_and(y_true == y_pred, y_true == 1)) / len(y_true)  # np.sum(y_true == 1)
    t_neg = np.sum(np.bitwise_and(y_true == y_pred, y_true == 0)) / len(y_true)  # np.sum(y_true == 0)
    f_pos = np.sum(np.bitwise_and(y_true != y_pred, y_true == 0)) / len(y_true)  # np.sum(y_true == 0)
    f_neg = np.sum(np.bitwise_and(y_true != y_pred, y_true == 1)) / len(y_true)  # np.sum(y_true == 1)

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap([
        [t_pos, f_neg],
        [f_pos, t_neg]
    ], annot=True, fmt='.2%', cmap='Blues')

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Actual class')

    plt.title(f'Confusion matrix of {model_type}')

    xticklabels = [f'Many {model_type}', f'Few {model_type}']
    ax.set_xticklabels(xticklabels)

    yticklabels = [f'Many {model_type}', f'Few {model_type}']
    ax.set_yticklabels(yticklabels)
    plt.show(block=False)


model = load_model(f'models/model.h5')
headline_preproc, sentiment = preprocess_headline('Warmup')
model.predict([headline_preproc, sentiment])

is_training = False
is_testing = False
is_my_testing = False
while True:
    inp = input(
        '\n\nDo you want to train or to test the model? (0-train, 1-test on your headlines, 2-test on the scraped data): ')
    if inp.isnumeric() and int(inp) in (0, 1, 2):
        if int(inp) == 0:
            is_training = True
            break
        elif int(inp) == 1:
            is_my_testing = True
            break
        elif int(inp) == 2:
            is_testing = True
            break
        else:
            print('Invalid option.')
    else:
        print('Invalid option.')

if is_training:
    # Plotting training history
    def plot_history(fit_history):
        plt.figure(figsize=(8, 6))
        plt.plot(fit_history.history['loss'], label='loss')
        plt.plot(fit_history.history['val_loss'], label='val_loss')
        plt.title(f'Loss during the training of the model')
        plt.legend()
        plt.figure(figsize=(8, 6))
        plt.plot((np.array(fit_history.history['out_views_acc']) + np.array(fit_history.history['out_likes_acc']) +
                  np.array(fit_history.history['out_dislikes_acc'])) / 3, label='acc')
        plt.plot(
            (np.array(fit_history.history['val_out_views_acc']) + np.array(fit_history.history['val_out_likes_acc']) +
             np.array(fit_history.history['val_out_dislikes_acc'])) / 3, label='val_acc')
        plt.title(f'Accuracy during the training of the model')
        plt.legend()
        plt.show(block=False)


    def get_callbacks():
        def lr_scheduler(epoch, lr):
            init_lr = 0.01
            cycle = 5
            min_lr = 1e-5

            if init_lr * (np.math.cos(np.pi / 2 / cycle * (epoch - cycle * (epoch // cycle)))) + min_lr < min_lr:
                lr = init_lr
            else:
                lr = init_lr * (np.math.cos(np.pi / 2 / cycle * (epoch - cycle * (epoch // cycle)))) + min_lr

            return lr

        fit_callbacks = [
            callbacks.LearningRateScheduler(
                lr_scheduler
            ),
            callbacks.ModelCheckpoint(
                monitor='val_out_views_acc',
                save_best_only=True,
                filepath=f'models/model.h5'
            )
        ]
        return fit_callbacks


    get_y()
    # Callbacks
    fit_callbacks = get_callbacks()

    # Training
    from model import build_model

    # Views predictive model
    model = build_model(MAXLEN, NUM_WORDS)
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics='acc')

    train_inputs = [X_headline_preproc_train, X_sentiment_train]
    train_outputs = [y_views_train, y_likes_train, y_dislikes_train]

    val_inputs = [X_headline_preproc_val, X_sentiment_val]
    val_outputs = [y_views_val, y_likes_val, y_dislikes_val]

    history = model.fit(train_inputs, train_outputs, batch_size=128, epochs=20,
                        validation_data=(val_inputs, val_outputs), callbacks=fit_callbacks)

    plot_history(history)

    # Comparing model's accuracy with the trivial baseline which is calculated
    # by assuming that every data element has a label of majority
    test_inputs = [X_headline_preproc_test, X_sentiment_test]
    test_outputs = [y_views_test, y_likes_test, y_dislikes_test]
    *_, acc_views, acc_likes, acc_dislikes = model.evaluate(test_inputs, test_outputs)

    y_pred_views, y_pred_likes, y_pred_dislikes = np.round(
        model.predict([X_headline_preproc_test, X_sentiment_test])).reshape((3, -1))

    print(f'{acc_views:.2%} - views model accuracy')
    print(
        f'{np.max((y_views_test.sum() / len(y_views_test), 1 - y_views_test.sum() / len(y_views_test))):.2%} - views baseline')

    print(f'{acc_likes:.2%} - likes model accuracy')
    print(
        f'{np.max((y_likes_test.sum() / len(y_likes_test), 1 - y_likes_test.sum() / len(y_likes_test))):.2%} - likes baseline')

    print(f'{acc_dislikes:.2%} - dislikes model accuracy')
    print(
        f'{np.max((y_dislikes_test.sum() / len(y_dislikes_test), 1 - y_dislikes_test.sum() / len(y_dislikes_test))):.2%} - dislikes baseline')

    confusion_matrix(y_views_test, y_pred_views, model_type='views')
    confusion_matrix(y_likes_test, y_pred_likes, model_type='likes')
    confusion_matrix(y_dislikes_test, y_pred_dislikes, model_type='dislikes')

    # Saving the tokenizer
    with open('models/tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

elif is_my_testing:
    print()
    while True:
        headline = input('Enter a headline you want to score:')
        score_to_word = lambda x: 'few' if x <= 0.5 else 'many'
        sentiment_to_word = lambda x: 'negative' if x <= 0 else 'positive'
        headline_preproc, sentiment = preprocess_headline(headline)
        views_pred, likes_pred, dislikes_pred = np.array(model.predict([headline_preproc, sentiment])).flatten()

        print(
            f'A video with such a headline is predicted to have {score_to_word(views_pred)} views, {score_to_word(likes_pred)} likes,'
            f' and {score_to_word(dislikes_pred)} dislikes. Sentiment of the headline is {sentiment_to_word(sentiment)}.\n')

elif is_testing:
    get_y()
    model = load_model(f'models/model.h5')

    # Comparing model's accuracy with a trivial baseline which is calculated
    # by setting every label of the data to the most frequent label
    test_inputs = [X_headline_preproc_test, X_sentiment_test]
    test_outputs = [y_views_test, y_likes_test, y_dislikes_test]
    *_, acc_views, acc_likes, acc_dislikes = model.evaluate(test_inputs, test_outputs)

    y_pred_views, y_pred_likes, y_pred_dislikes = np.round(
        model.predict([X_headline_preproc_test, X_sentiment_test])).reshape((3, -1))

    print(f'{acc_views:.2%} - views model accuracy')
    print(
        f'{np.max((y_views_test.sum() / len(y_views_test), 1 - y_views_test.sum() / len(y_views_test))):.2%} - views baseline')

    print(f'{acc_likes:.2%} - likes model accuracy')
    print(
        f'{np.max((y_likes_test.sum() / len(y_likes_test), 1 - y_likes_test.sum() / len(y_likes_test))):.2%} - likes baseline')

    print(f'{acc_dislikes:.2%} - dislikes model accuracy')
    print(
        f'{np.max((y_dislikes_test.sum() / len(y_dislikes_test), 1 - y_dislikes_test.sum() / len(y_dislikes_test))):.2%} - dislikes baseline')


    confusion_matrix(y_views_test, y_pred_views, model_type='views')
    confusion_matrix(y_likes_test, y_pred_likes, model_type='likes')
    confusion_matrix(y_dislikes_test, y_pred_dislikes, model_type='dislikes')
