import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

import re

nltk.download('stopwords')


def load_data(database_filepath):
    
    '''
    Loads cleaned disaster data from the database and
    extracts arrays with dependent and independent variables
    
    INPUTS
        - database_filepath (str): location of the database
    
    OUTPUTS
        - X (array): independent variables
        - Y (array): dependent variables
        - category_names (list): names of all posible disaster categories
    
    ''' 
    
    # read from data base
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', con = engine)
    
    # extract the category names from the dataset
    category_names = list(df.drop(['id', 'message', 'original', 'genre', 'related'], axis = 1).columns)

    # arrays with dependent and independent variables
    X = df[['message']].values
    Y = df.drop(['id', 'message', 'original', 'genre', 'related'], axis = 1).values
    
    return X, Y, category_names


def tokenize(text):
    
    
    ''' One word tokenizer from a string
    
    INPUTS
        - text (str): a text string
    
    OUTPUT
        - text_tokenized (str): one word tokenized version of text in list form
    ''' 
    
    # clean text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    text_tokenized = word_tokenize(text)
    
    # remove stop words
    text_tokenized = [word for word in text_tokenized if word not in stopwords.words('english')]
    
    return text_tokenized


def build_model():
    
    '''
    Creates a pipeline for the model which predicts 
    category disasters
    
    OUTPUT
    - model (pipeline): Pipeline of a RandomForest clasifier
    uning Tfidf transformations over text variables
        
    ''' 
    
    model = Pipeline([
    ('count', CountVectorizer(tokenizer = tokenize)), 
    ('tfidf', TfidfTransformer()), 
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Predicts and evaluate a model previously train printing
    a confusion matrix for each category with accuracy, precision
    and recall metrics
    
    INPUTS
        - model (pipeline): pipeline of a multiclassifier trained model
        - X_test (array): test set for the independent variables of the model
        - Y_test (array): test set for the dependent variables of the model
        - category_names (list): names of all posible disaster categories
    ''' 
    
    
    Y_pred = model.predict(X_test.ravel())
    
    # evaluation of every category independently
    for col in range(len(category_names)):
        print(category_names[col])
        print(classification_report(Y_test[:, col], Y_pred[:, col]))
    
    


def save_model(model, model_filepath):
    
    '''
    Saves the fitted model in pickle format
    
    INPUTS
        - model (pipeline): fitted multiclissifier model
        - model_filepath (string): path where the model will be saved
    ''' 

    
    with open( model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.ravel(), Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()