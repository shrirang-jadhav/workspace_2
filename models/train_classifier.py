# import libraries

import sys
import pickle
import re
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Loads data from database
    Args:
        database_filepath: path to database
    Returns:
        X(DataFrame) : feature
        Y(DataFrame) : labels
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('dstr_rspns_data', con=engine)
    
    X = df[['message']].values[:, 0]
    y = df.drop(['id','message','original','genre'], axis=1)
    
    category_names = df.columns[4:]
    
    return X, y, category_names


def tokenize(text):
    """
    Tokenizes a given text.
    Args:
        text: text string
    Returns:
        (str[]): array of clean tokens
    """
    # normalize first
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # then tokenize
    tokens = word_tokenize(text)

    # remove stop words
    words = [w for w in tokens if w not in stopwords.words("english")]

    # initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize and strip
    clean_tokens = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).strip()
        clean_tokens.append(clean_word)

    return tokens


def build_model():
    """Builds classification model """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [5] }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2,  verbose=3,n_jobs=5)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model against a test dataset
    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: String array of category names
    """
    y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
        print(classification_report(Y_test.values[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the model to a Python pickle
    Args:
        model: Trained model
        model_filepath: Path where to save the model which we will access in run.py later
    """
    with open(model_filepath, 'wb') as file:
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