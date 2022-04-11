from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import os
import pandas as pd
import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import string
import re
import requests

PUNCT_TO_REMOVE = string.punctuation+'“”’'

def remove_punctuation(text):
    return text.translate(str.maketrans(PUNCT_TO_REMOVE, ' '*len(PUNCT_TO_REMOVE)))

#STOPWORDS = set(nltk.corpus.stopwords.words('english')+['','bought','buy','fit'])
stopwords_list = requests.get(
    "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
STOPWORDS = set(stopwords_list.decode().splitlines()+['','bought','buy','fit','good']) 

def remove_stopwords(text):
    return " ".join([word for word in str(
        text).split() if word.lower() not in STOPWORDS])

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


EMOTICONS = np.load('/home/admin/topic-classifier/utils/emoticons.npy', allow_pickle='TRUE').item()


def remove_emoticons(text):
    emoticon_pattern = re.compile(
        u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

with open('./google-10000-english/google-10000-english-no-swears.txt', 'r') as f:
    top1000 = f.readlines()
top1000 = [i.strip() for i in top1000]
top1000 = top1000[:1000]
top1000_set = set(top1000)

def remove_top_1000_google(text):
    return " ".join([word for word in str(
        text).split() if word not in top1000_set])

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def remove_line_breaks(text):
    return ' '.join(text.splitlines())



def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_subreddit_symbol(text):

    return text.replace('/r/', '')

def remove_double_space(text):
    
    return re.sub('\s+', ' ', text)

def tokenize_(text):
    return ' '.join(word_tokenize(text))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemma_(text):
    text = text.split(' ')
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

def main_preprocessor(text):
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
    text = remove_stopwords(text)
    text = lemma_(text)
    text = remove_stopwords(text)
    #text = remove_top_1000_google(text)
    
    return text


def main(df, col1, col2):
    
    with mp.Pool(mp.cpu_count()) as pool:
        df[col2] = pool.map(main_preprocessor, df[col1])
    print('text preprocessed')

    #df = df.drop_duplicates()
    #df = df[df.topic.isna() == False]
    #df = df[df.text.isna() == False]

    return df