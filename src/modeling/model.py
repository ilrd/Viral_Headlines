from tensorflow.keras.layers import *
from tensorflow.keras import Model


def build_model(MAXLEN, NUM_WORDS):
    DIM = 256  # Dimension of the embedding space

    inp_sentiment = Input(shape=(1,))
    hid_sentiment = RepeatVector(4)(inp_sentiment)
    hid_sentiment = Flatten()(hid_sentiment)

    inp_headline_preproc = Input(shape=(MAXLEN,))
    hid = Embedding(NUM_WORDS, DIM)(inp_headline_preproc)

    hid = LSTM(64, return_sequences=True)(hid)
    hid = Flatten()(hid)
    hid = Dense(64, activation='relu')(hid)
    hid = Dropout(0.5)(hid)
    hid = Dense(16, activation='relu')(hid)
    hid = Concatenate()([hid, hid_sentiment])
    hid = Dense(64, activation='relu')(hid)
    hid = Dense(32, activation='relu')(hid)

    out_views = Dense(1, activation='sigmoid', name='out_views')(hid)
    out_likes = Dense(1, activation='sigmoid', name='out_likes')(hid)
    out_dislikes = Dense(1, activation='sigmoid', name='out_dislikes')(hid)

    model = Model([inp_headline_preproc, inp_sentiment], [out_views, out_likes, out_dislikes])

    return model
