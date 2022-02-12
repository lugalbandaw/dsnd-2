import sys

# import libraries
import pandas as pd

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    '''
    load_data   function to load data from database
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM lugalTABLE", engine)
    X = df['message']
    Y = df.drop(['message', 'original', 'genre'], axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    tokenize   function to tokenize text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model   function to build model pipeline for ML using GridSearch
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidi', TfidfTransformer()),
        ('moc', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'moc__estimator__n_estimators': [25, 50],
        'moc__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evalualte_model   function to evaluate fitted model on test data
    '''
    # make prediction
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)

    # iterate through the columns and call sklearn's classification_report on each
    for target in category_names:
        cr = classification_report(Y_test[target], Y_pred[target])
        print('\n*** '+target+'***\n')
        print(cr)


def save_model(model, model_filepath):
    '''
    save_model   function to save fitted model using pickle
    '''
    # open file to save model
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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