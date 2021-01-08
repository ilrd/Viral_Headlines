import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.getcwd())


class News:
    def __init__(self, news_df, scraped_date=None):
        self.headlines = news_df.loc[:, 'headline']
        self.authors = news_df.loc[:, 'author']
        self.views = news_df.loc[:, 'views']
        self.likes = news_df.loc[:, 'likes']
        self.dislikes = news_df.loc[:, 'dislikes']
        self.dates = news_df.loc[:, 'date']
        if scraped_date is not None:
            self.scraped_date = datetime.strptime(scraped_date, '%Y-%m-%d')

    def get_df(self):
        return pd.DataFrame(data=zip(self.headlines, self.authors, self.views, self.likes, self.dislikes, self.dates),
                            columns=['headline', 'author', 'views', 'likes', 'dislikes', 'date'])

    def __str__(self):
        return self.get_df().__repr__()

    def __repr__(self):
        return self.get_df().__repr__()

    def concat(self, news):
        return News(pd.concat([self.get_df(), news.get_df()], ignore_index=True))

    @staticmethod
    def SMA(points, window=5):
        sma_points = points.rolling(window, min_periods=1).mean()
        return sma_points

    def preprocess_views(self):
        # Setting 0s to missing values
        clean_views = []
        for view in self.views:
            if view == 0:
                view = None
            clean_views.append(view)
        self.views = pd.Series(clean_views)

        # Author-wise scaling and normalization
        for author in self.authors.unique():
            author_views = self.views[self.authors == author]
            views_rolled = self.SMA(author_views, 30)
            self.views[self.authors == author] = author_views / views_rolled

        scaler = StandardScaler()
        self.views = pd.Series(scaler.fit_transform(np.log(self.views).values.reshape(-1, 1)).flatten())

    def preprocess_likes(self):
        # Setting 0s to missing values
        clean_likes = []
        for likes in self.likes:
            if likes == 0:
                likes = None
            clean_likes.append(likes)
        self.likes = pd.Series(clean_likes)

        # Author-wise scaling and normalization
        for author in self.authors.unique():
            author_views = self.likes[self.authors == author]
            views_rolled = self.SMA(author_views, 30)
            self.likes[self.authors == author] = author_views / views_rolled

        scaler = StandardScaler()
        self.likes = pd.Series(scaler.fit_transform(np.log(self.likes).values.reshape(-1, 1)).flatten())

    def preprocess_dislikes(self):
        # Setting 0s to missing values
        clean_dislikes = []
        for dislikes in self.dislikes:
            if dislikes == 0:
                dislikes = None

            clean_dislikes.append(dislikes)
        self.dislikes = pd.Series(clean_dislikes)

        # Author-wise scaling and normalization
        for author in self.authors.unique():
            author_views = self.dislikes[self.authors == author]
            views_rolled = self.SMA(author_views, 30)
            self.dislikes[self.authors == author] = author_views / views_rolled

        scaler = StandardScaler()
        self.dislikes = pd.Series(scaler.fit_transform(np.log(self.dislikes).values.reshape(-1, 1)).flatten())

    def preprocess_dates(self):
        clean_dates = []
        for date in self.dates:
            if type(date) == str:
                raw_date = date.strip()

                # Matching "Day Month Year" as "17 Dec 2018"
                if re.match(r'\d+ \w\w\w\w? \d\d\d\d', raw_date):
                    if 'Sept' in raw_date:
                        raw_date = raw_date.replace('Sept', 'Sep')
                    clean_dates.append((self.scraped_date - datetime.strptime(raw_date, '%d %b %Y')).days)

                # Setting date to None by default
                else:
                    clean_dates.append(None)
            else:
                clean_dates.append(date)

        self.dates = pd.Series(clean_dates)

        return pd.Series(self.dates)


def connect_data(preprocess=False):
    columns = ['headline', 'author', 'views', 'likes', 'dislikes', 'date']
    all_news = News(pd.DataFrame(columns=columns))
    file_names = os.listdir('/home/ilolio/PycharmProjects/Viral_Headlines/data/raw/')
    file_names.remove('used_links.txt')
    paths = list(map(lambda x: '/home/ilolio/PycharmProjects/Viral_Headlines/data/raw/' + x, file_names))
    for path in paths:
        df = pd.read_csv(path)
        news = News(df, path.split('/')[-1][:-4])
        if preprocess:
            news.preprocess_dates()
            news.preprocess_views()
            news.preprocess_likes()
            news.preprocess_dislikes()
        all_news = all_news.concat(news)

    return all_news


news_df = connect_data(preprocess=True).get_df()
news_df.dropna(axis=0, inplace=True)
news_df.drop_duplicates(subset='headline', inplace=True)

# Text preprocessing
from nlp_preprocessing import NLP_Preprocessor

nlp_preprocessor = NLP_Preprocessor()
headline = news_df.loc[:, 'headline']

print('Headline preprocessing started')
text_data = nlp_preprocessor.split_sent(headline)
tokens = nlp_preprocessor.tokenize(text_data)
tokens = nlp_preprocessor.lowercase(tokens)
tokens = nlp_preprocessor.lemmatize_tokens(tokens)
tokens = nlp_preprocessor.remove_stopwords(tokens)
tokens = nlp_preprocessor.remove_punctuation(tokens)
text_data = nlp_preprocessor.tokens_to_text(tokens)

news_df['headline_preproc'] = text_data
print('Headline preprocessing finished')

# Sentiment analysis
print('Sentiment analysis started')

news_df['sentiment'] = nlp_preprocessor.get_sentiments(headline)

print('Sentiment analysis finished')

# Saving the data
# news_df.to_csv('data/preprocessed/news.csv', index=False)
