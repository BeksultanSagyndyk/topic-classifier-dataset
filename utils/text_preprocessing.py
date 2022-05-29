from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from pymystem3 import Mystem
mystem = Mystem()

import numpy as np
import multiprocessing as mp
import string
import re
import requests
import pandas as pd
import nltk

#punctuations
PUNCT_TO_REMOVE = string.punctuation + '“”’'

# en stopwords
# STOPWORDS = set(nltk.corpus.stopwords.words('english')+['','bought','buy','fit'])
PATH = "https://gist.githubusercontent.com\
/rg089/35e00abf8941d72d419224cfd5b5925d\
/raw/12d899b70156fd0041fa9778d657330b024\
b959c/stopwords.txt"
stopwords_list = requests.get(PATH).content
english_stopwords = set(stopwords_list.decode().splitlines() + ['', 'bought', 'buy', 'fit', 'good'])

# russian stopwords
russian_stopwords = set(nltk.corpus.stopwords.words('russian'))

#emoticons
EMOTICONS = np.load('./emoticons.npy',
                    allow_pickle='TRUE').item()

#top search words google
with open('./google-10000-english/google-10000-english-no-swears.txt', 'r') as f:
    top1000 = f.readlines()
top1000 = [i.strip() for i in top1000]
top1000 = top1000[:1000]
top1000_set = set(top1000)

def remove_punctuation(text: str) -> str:
    '''
    This function removes !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”’ from input text
    :param text: str
    :return: str
    '''

    return text.translate(str.maketrans(PUNCT_TO_REMOVE, ' ' * len(PUNCT_TO_REMOVE)))

def remove_stopwords(text: str, stopwords: set) -> str:
    '''
    This function removes stopwords from input text
    :param text: str
    :return: str
    '''
    return " ".join([word for word in str(
        text).split() if word.lower() not in stopwords])


def remove_emoji(text: str) -> str:
    '''
    This function removes some emoticons,symbols & pictographs,
    transport & map symbols, flags from input text data
    :param text: str
    :return: str
    '''
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_emoticons(text: str) -> str:
    '''
    This function removes emoticons from input data
    :param text: str
    :return: str
    '''
    emoticon_pattern = re.compile(
        u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)


def remove_top_1000_google(text: str) -> str:
    '''
    This function removes most frequent words from google queries
    top 1000
    :param text: str
    :return: str
    '''
    return " ".join([word for word in str(
        text).split() if word not in top1000_set])


def remove_html(text: str) -> str:
    """
    This function removes html attributes from input text data
    :param text: str
    :return: str
    """
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def remove_line_breaks(text: str) -> str:
    """
    This function removes line breaks from input text data
    :param text: str
    :return: str
    """
    return ' '.join(text.splitlines())


def remove_urls(text: str) -> str:
    """
    This function removes web links from input text data
    :param text: str
    :return: str
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_subreddit_symbol(text: str) -> str:
    """
    This function removes special subbredit symbol '/r'
    from input text data
    :param text: str 
    :return: str
    """
    return text.replace('/r/', '')


def remove_double_space(text):
    """
    This function removes double spaces from input text data
    :param text: str
    :return: str
    """
    return re.sub('\s+', ' ', text)


def tokenize_(text: str) -> str:
    """
    This function tokenize input text data
    by using nltk library
    :param text: str
    :return: str
    """
    return ' '.join(word_tokenize(text))


def lemm(text: str, lang: str) -> str:
    """
    This function lemmatize every token from input text data
    :param text: str
    :lang: str, 'en' or 'ru'
    :return: str
    """
    if lang == 'en':
        text = text.split(' ')
        text = [lemmatizer.lemmatize(word) for word in text]
    if lang == 'ru':
        text = mystem.lemmatize(text)
    text = ' '.join(text)
    return text

def main_preprocessor(text):
    """
    This function does whole prerocessing operations with input text data
    :param text: str
    :return: str
    """

    text = remove_subreddit_symbol(text)
    text = remove_line_breaks(text)
    text = remove_double_space(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_emoticons(text)
    text = remove_emoji(text)

    text = remove_punctuation(text)

    text = text.lower()
    text = tokenize_(text)
    text = remove_stopwords(text,english_stopwords)
    text = lemm(text,'en')
    text = remove_stopwords(text)
    # text = remove_top_1000_google(text)

    return text


def main(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    This function applies main preprocessing function to column with text data from pandas dataframe 
    :param df: pd.DataGrame
    :param col1: name of column, str
    :param col2: name of column, str
    :return: pd.DataFrame
    """

    with mp.Pool(mp.cpu_count()) as pool:
        df[col2] = pool.map(main_preprocessor, df[col1])
    print('text preprocessed')

    return df
