import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import nltk
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


class Toolbox:
    def __init__(self,):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.punctuations = string.punctuation
        self.tfidf = TfidfVectorizer(
                    analyzer='word',
                    tokenizer=self.dummy_fun,
                    preprocessor=self.dummy_fun,
                    token_pattern=None)
        self.ordinal_encoder = OrdinalEncoder()
        self.oneHot_encoder = OneHotEncoder()
        nltk.download('stopwords')

    @staticmethod
    def dummy_fun(doc):
        return doc

    def preprocess(self, input_text: str) -> list:
        tokenized_list = self.nlp(input_text)
        tokenized_list = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-"
                          else word.lower_ for word in tokenized_list]
        tokenized_list = [word for word in tokenized_list if word not in self.stop_words
                          and word not in self.punctuations]
        return tokenized_list

    def process_df(self, df_in: DataFrame, to_vectorized: list, to_oneHot: list = None) -> DataFrame:
        vector_target = []
        oneHot_target = []
        i = 0
        df = df_in.copy()
        for item in to_vectorized:
            vector_target = vector_target + df[item].tolist()
        for item in to_oneHot:
            if i == 0:
                oneHot_target = df[item].to_numpy().reshape(-1, 1)
                i += 1
            else:
                oneHot_target = np.vstack([oneHot_target, df[item].to_numpy().reshape(-1, 1)])

        self.tfidf.fit(vector_target)
        self.oneHot_encoder.fit(oneHot_target)

        encoded = []
        for item in to_oneHot:
            if item == to_oneHot[0]:
                encoded = self.oneHot_encoder.transform(df[item].to_numpy().reshape(-1, 1))
            else:
                encoded.append(self.oneHot_encoder.transform(df[item].to_numpy().reshape(-1, 1)))
        encoded = encoded
        encoded_df = pd.DataFrame(data=encoded, columns=to_oneHot)

        for tuple in df.itertuples():
            index = tuple[0]
            for item in to_vectorized:
                num = df.columns.get_loc(item)
                df.at[index, item] = self.tfidf.transform([tuple[num]]).toarray()[0][np.newaxis, :]
            for item in to_oneHot:
                df.at[index, item] = encoded_df[item][index].toarray()

        return df
