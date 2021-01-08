import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import wordnet
from string import punctuation
from collections import Counter
import numpy as np
from transformers import pipeline


class NLP_Preprocessor:

    def __init__(self):
        self.classifier = pipeline('sentiment-analysis')
        self.classifier("Initialization")

    # Splitting sentences
    @staticmethod
    def split_sent(text_data):
        # Sentence splitting
        sent_tok_text_data = [' [SENT] '.join(sent_tokenize(text)) for text in text_data]

        return sent_tok_text_data

    # Tokenization
    @staticmethod
    def lowercase(tokens):
        tokens = [[token.lower() for token in text] for text in tokens]

        return tokens

    # Tokenization
    @staticmethod
    def tokenize(text_data, num_words=None):
        tokens = [word_tokenize(text) for text in text_data]

        if num_words:
            counter = Counter(word_tokenize(' '.join(text_data)))
            common_words = [word for word, _ in counter.most_common(num_words)]

            tokens = [[token for token in text if token in common_words] for text in tokens]

        return tokens

    # Lemmatization
    @staticmethod
    def lemmatize_tokens(tokens, pos=True):
        word_lem = WordNetLemmatizer()

        if pos:
            return [[word_lem.lemmatize(word, pos) for word, pos in
                     zip(tokens_i, map(NLP_Preprocessor.wordnet_pos, tokens_i))] for tokens_i in tokens]
        else:
            return [[word_lem.lemmatize(word) for word in tokens_i] for tokens_i in tokens]

    # POS tagging
    @staticmethod
    def wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

        return tag_dict.get(tag, wordnet.NOUN)

    # Stemming
    @staticmethod
    def stem_lokens(tokens):
        sst = EnglishStemmer()
        return [[sst.stem(word) for word in tokens_i] for tokens_i in tokens]

    # Stopwords removal
    @staticmethod
    def remove_stopwords(tokens):
        stops = stopwords.words('english')

        return [[word for word in tokens_i if word not in stops] for tokens_i in tokens]

    # Punctuation removal
    @staticmethod
    def remove_punctuation(tokens):
        return [[word for word in tokens_i if word not in punctuation] for tokens_i in tokens]

    # Tokens to text
    @staticmethod
    def tokens_to_text(tokens):
        return [' '.join(tokens_i) for tokens_i in tokens]

    # Calculating news sentiments
    def get_sentiments(self, text_data):
        X_sentiment = []
        for i in range(len(text_data) // 128 + 1):
            X_sentiment += list(map(lambda x: x['score'] if x['label'].lower() == 'positive' else -x['score'],
                                    self.classifier(list(text_data)[128 * i:128 * (i + 1)])))
            if len(text_data) != 1:
                print(f'{i + 1} of {len(text_data) // 128 + 1} batches are done', end='\r', flush=True)
        X_sentiment = np.array(X_sentiment)
        return X_sentiment