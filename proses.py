# preprocess.py

import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import BorderlineSMOTE

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def upload_dataset(file):
     if file is not None:
        file_name = file.name if file.name is not None else "Unknown File"
        dataset = pd.read_csv(file)
        return dataset, file_name
     return None, None

def preprocess(text):
    # Remove URL
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    # Remove emoji
    text = re.sub("["
                  u"\U0001F600-\U0001F64F"  # emoticons
                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                  u"\U00002702-\U000027B0"
                  u"\U000024C2-\U0001F251"
                  "]+", '', str(text), flags=re.UNICODE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', str(text))
    # Case folding
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

def tfidf(df, vocabulary=None):
    tfidf = TfidfVectorizer(vocabulary=vocabulary)
    tfidf_matrix = tfidf.fit_transform(df['text'])

    tfidf_vect_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    df_tfidf = pd.concat([df, tfidf_vect_df], axis=1)
    df_tfidf = df_tfidf.drop(['text'], axis=1)
    df_tfidf = df_tfidf.dropna()

    return df_tfidf

def borderline_smote(X_train, y_train):
    bsm = BorderlineSMOTE()
    X_train_res, y_train_res = bsm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

def load_model(algortima, c):
    if c == 0.1 and algortima == "SVM + Borderline-SMOTE":
        model_path = "model/svc_c01bs.joblib"
    elif c == 1.0 and algortima == "SVM + Borderline-SMOTE":
        model_path = "model/svc_c1bs.joblib"
    elif c == 0.1 and algortima == "SVM":
        model_path = "model/svc_c01.joblib"
    elif c == 1.0 and algortima == "SVM":
        model_path = "model/svc_c1.joblib"
    else:
        raise ValueError("Kombinasi nilai C dan algortima tidak dikenali.")
    
    return joblib.load(model_path)



