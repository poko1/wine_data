import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics
import math
from pathlib import Path
import pickle
import logging
import matplotlib.pyplot as plt
from numpy import absolute,mean,std

# vectorization
def vectorizing(X_train, X_test, y_train, y_test): # save the vocabulary
    vectorizer = CountVectorizer(max_features=4000,ngram_range=(1,2)) #ngram_range = (1, 2), min_df=10
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    X_train.toarray()
    X_test.toarray()
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    pickle.dump(vectorizer, open(Path('model','vectorizer_all.pickle'), 'wb')) #Save vectorizer

    return X_train, X_test, y_train, y_test

# train model
from sklearn.svm import SVR
def train_test_model(X_train, X_test, y_train, y_test): #and validate
    clf = MultiOutputRegressor(RandomForestRegressor(max_depth=10, random_state=0))
    #clf = MultiOutputRegressor(SVR(epsilon=0.2))
    #gbr = GradientBoostingRegressor(
    #n_estimators=20, 
    #learning_rate=0.1, 
    #max_depth=10)
    #clf = MultiOutputRegressor(estimator=gbr)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    train_rmse = metrics.mean_squared_error(y_train, y_train_pred, squared=False)
    #train_rmse = math.sqrt(mse)
    print(train_rmse)

    y_test_pred = clf.predict(X_test)
    test_rmse = metrics.mean_squared_error(y_test, y_test_pred, squared=False)
    #test_rmse = math.sqrt(mse)
    print(test_rmse)
    score_rmse = metrics.mean_squared_error(y_test[:,0], y_test_pred[:,0], squared=False)
    price_rmse = metrics.mean_squared_error(y_test[:,1], y_test_pred[:,1], squared=False)
    

    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test_pred[:,0])), y_test_pred[:,0], color='green')
    plt.scatter(range(len(y_test_pred[:,0])), y_test[:,0], color='red')
    plt.title("Comparison of Actual and Predicted Score", fontsize=18)
    plt.savefig('score_final.png')

    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test_pred[:,1])), y_test_pred[:,1], color='green')
    plt.scatter(range(len(y_test_pred[:,1])), y_test[:,1], color='red')
    plt.title("Comparison of Actual and Predicted Score", fontsize=18)
    plt.savefig('price_final.png')


    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(clf, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    # force the scores to be positive
    n_scores = absolute(n_scores)
    # summarize performance
    print('RMSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    # log model result
    with open('model/model_log_all.txt', 'w') as f:
        f.write('RMSE Train: ')
        f.write(str(train_rmse))
        f.write(' RMSE Test: ')
        f.write(str(test_rmse))
        f.write(' RMSE Test Score: ')
        f.write(str(score_rmse))
        f.write(' RMSE Test Price: ')
        f.write(str(price_rmse))
        f.write(' CV RMSE Train: ')
        f.write(str(mean(n_scores)))
        f.write(' CV RMSE Train Std: ')
        f.write(str(std(n_scores)))

    if (abs(train_rmse - test_rmse)<5):
        filename = 'finalized_model_all.sav'
        pickle.dump(clf, open(Path('model',filename), 'wb'))
    else: 
        logging.info('Model training and testing RMSE has crossed threshold')
    return clf



if __name__ == '__main__':

    # load data from final directory 
    df = pd.read_csv("final_data/final.csv")
    # for now let us only use text description
    X = df['description']
    y = df[['points','price']]
    # print(X.head())
    print(y.head())
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104,test_size=0.25, shuffle=True)
    # vectorize the data
    X_train, X_test, y_train, y_test = vectorizing(X_train, X_test, y_train, y_test)
    #print(X_train)
    #print(y_train)
    # train the model (can do cross validation and hyperparameter tuning later)
    train_test_model(X_train, X_test, y_train, y_test)
    # save model if certain threshold is met (by versioning and during loading, load the latest: for now just add a date or overwrite)   
    

