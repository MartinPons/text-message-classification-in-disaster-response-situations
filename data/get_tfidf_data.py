# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import json
import plotly
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline

engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

categories_df = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

cat_n_messages = categories_df.sum().sort_values(ascending = False)

category_names = list(cat_n_messages.index)
category_names = list(map(lambda x: x.replace('_', ' ').upper(), category_names))
category_counts = cat_n_messages.values

cat_names = list(categories_df.columns)

messages_dict = dict.fromkeys(cat_names)
messages_dict = {key:"" for (key, value) in messages_dict.items()}

for cat in cat_names:
    for row in range(df.shape[0]):
        
        if df.iloc[row][cat] == 1:
            messages_dict[cat] = messages_dict[cat] + " " + df.iloc[row]["message"]


messages_list = [value for value in messages_dict.values()]
messages_sample = messages_list[0:2]


def tokenize(text):
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = re.sub(r"[0-9]", "", text.lower())

    text_tokenized = word_tokenize(text)
    
     
    # text = [word.lower() for word in text]
    text_tokenized = [word for word in text_tokenized if word not in stopwords.words('english')]
    
    return text_tokenized



tf = Pipeline([('count', CountVectorizer(tokenizer = tokenize)), 
                 ('tfidf', TfidfTransformer())])



vect = CountVectorizer(tokenizer = tokenize)
vect_fitted = vect.fit_transform(messages_list)

tfidf = TfidfTransformer()
tfidf_matrix = tfidf.fit_transform(vect_fitted)

# tf.TfidfTransformer.get_feature_names()

tf_array = tfidf_matrix.toarray()
# tf_df = pd.DataFrame(tf_array, columns = vect.get_feature_names())

import pickle
filehendler = open("tfidf_array.pickle", 'wb')
pickle.dump(tf_array, filehendler)


filehendler = open("vect_fitted.pickle", 'wb')
pickle.dump(vect, filehendler)


grams = vect.get_feature_names()


top_words = set([])

for row in range(tf_array.shape[0]):
        
        # row order index
        order_index = np.argsort(tf_array[row, ])[0:3]
        
        top_words.update(np.array(grams)[order_index])


top_word_index = [idx for idx in range(len(grams)) if grams[idx] in top_words]

tf_array_filtered = tf_array[:, top_word_index]

from plotly.graph_objects import Heatmap
import plotly.graph_objects


from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
init_notebook_mode()

trace0 = Heatmap(x = list(top_words),  
                 y = grams[0:75], 
                 z = tf_array_filtered, 
                 type = "heatmap", 
                 colorscale = "viridis")


# ESTAN POR ORDEN ALFABETICO!!!!

data = [trace0]
layout = Layout(
    showlegend=False,
    height=600,
    width=600,
)

fig = dict( data=data, layout=layout )

plot(fig)  


plotly.offline.plot({
"data": [Heatmap(
        x = grams[0:2],
        y = list(top_words), 
        z = tf_array_filtered,
        type='heatmap')]})

plotly.offline.plot({
"data": [
    plotly.graph_objs.Scatter(    x = [1, 2, 3],
    y = [4, 5, 6])]})




# LEMATIZO, PILLO SOLO EL TOP 10 PARA CADA CATEGORIA, Y CON LO QUE 
# ME SALGA MIRO DE HACER EL HEATMAP


import pickle
filehendler = open("tfidf_matrix_categories.pickle", 'wb')
pickle.dump(tfidf_matrix, filehendler)

filehendler = open("tfidf_grams.pickle", 'wb')
pickle.dump(tfidf_matrix, filehendler)

filehandler = open("tfidf_matrix_categories.pickle", 'rb')
pr = pickle.load(filehandler)

import plotly




# como hacer un heatmap
# https://www.tutorialspoint.com/plotly/plotly_heatmap.htm
