import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(messages_filepath, categories_filepath):
    
    '''Loads messages and categories data and merges them into a 
    single data frame joining by the id column
    
    INPUTS
        - messages_filepaths (str): file path where the messages csv data is located
        - categories_filepath (str): file path where the categories csv data is located
        
    OUTPUT
        - Merged data frame with messages and a categories column
        
    '''  
    # load csv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how = "left", on = "id")
    
    return df


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


def clean_data(df):
    
    '''Cleans the categories column from DataFrame obtained in "load_data" expanding it 
    to one column per category and reconverting the values in the new categories column into binary ones
    
    
    INPUTS
        - df (DataFrame): pandas DataFrame returned by the "load_data" function
    
    OUTPUTS
        - df: DataFrame with the categories expanded to one category per column and in binary mode.
    '''
    
    # split categories
    categories = df.categories.str.split(pat = ";", expand = True)
    
    # get category names
    row = categories.iloc[1,:]
    category_colnames = list(row.str.slice(stop = -2))
    categories.columns = category_colnames
    
    # reformat category data into binary values
    for column in categories:
        categories[column] = categories[column].str.slice(start = -1)
        categories[column] = pd.to_numeric(categories[column])
    
    # insert the categories in the main DataFrame
    df.drop("categories", axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    
    # remove duplicates
    duplicates = df.duplicated()
    df = df.loc[~duplicates]
    
    return df




def save_data(df, database_filename):
    ''' 
    Saves cleaned data obtain in the 'clean_data' function into a database in SQL format
    
    IMPUTS
        - df (DataFrame): DataFrame returned by the clean_data function
        - database_filename (str): path of the database where the table will be saved
    '''
    
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()