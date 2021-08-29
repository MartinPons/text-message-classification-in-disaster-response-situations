import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class TextLength(BaseEstimator, TransformerMixin):
    
    '''Transformer class to compute the text lenght of every message 
    in the model pipeline
    
    INPUTS
        - BaseEstimator (fun): provides the stimator with the basic parameters related methods
        - TrasnformerMixin (fun): adds fit_transform method
    
    OUTPUT:
        - Transformer which computes the text length in the message column
    ''' 
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):

        return pd.Series(X).apply(lambda x: len(x)).values.reshape(-1, 1)
    

class ExclamationPoints(BaseEstimator, TransformerMixin):
    
    '''Transformer class to check if there is at least one exclamation mark in the message
        
        INPUTS
            - BaseEstimator (fun): provides the stimator with the basic parameters related methods
            - TrasnformerMixin (fun): adds fit_transform method
    
        OUTPUT:
            - Transformer returning a column of zeroes and ones indicating the presence of exclamation points
    ''' 

    
    def fit(self, X, y = None):
        
        return self
    
    def transform(self, X):
        
        return pd.Series(X).apply(lambda x: ('!' in x) * 1).values.reshape(-1, 1)

    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # Genres plot data extraction
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Category plot data extraction
    categories_df = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    cat_n_messages = categories_df.sum().sort_values(ascending = True)
    
    # labels extraction and formatting
    category_names = list(cat_n_messages.index)
    category_names = list(cat_n_messages.index)
    category_names = list(map(lambda x: x.replace('_', ' ').upper(), category_names))
    
    # values
    category_counts = cat_n_messages.values
    
    # Text length histogram data extraction
    text_length = list(df.message.apply(lambda x: len(x)).values)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [Bar(
                x = category_counts,
                y = category_names,
                orientation = 'h'
            )
                    ], 
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "",
                    'tickfont': {'size': 8}
                },
                'height':600,
 
                'xaxis': {
                    'title': "Genre"
                }
                }
            }, 
        
        {
            'data': [Histogram(
                x = text_length
            )
                    ], 
            'layout': {
                'title': 'Distribution of text length (limited to messages from 1 to 600 characters)', 
                'xaxis': {
                    'range': [0, 600]
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()