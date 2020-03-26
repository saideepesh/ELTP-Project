"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Mohamed El Hajji
        (2) Sai Deepesh Pokala
        (3) Ibrahim Chiheb
        etc.
"""

"""
    Import necessary packages
"""
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
"""
        etc.
"""

"""
    Your methods implementing the models.
    
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
"""


def model_decision_tree(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = DecisionTreeClassifier() # default parameters gave the best results
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return accuracy, f1

def model_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = RandomForestClassifier(n_estimators = 750, bootstrap=True, class_weight = 'balanced') # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return accuracy, f1

def model_adaboost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = AdaBoostClassifier(n_estimators = 500) # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return accuracy, f1


def model_extra_trees(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = ExtraTreesClassifier(n_estimators = 1000, bootstrap = True) # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return accuracy, f1

def model_bagging(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = BaggingClassifier(n_estimators = 500) # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return accuracy, f1


def model_xgboost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = XGBClassifier(n_estimators = 1000)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return accuracy, f1

def model_catboost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = CatBoostClassifier(n_estimators = 25000,task_type = 'GPU', devices='0:1', early_stopping_rounds=50, max_depth=8, learning_rate=0.1)
    clf.fit(X_train, y_train, eval_set = (X_val, Y_val))

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return accuracy, f1

name2models = {'DecisionTreeClassifier' : model_decision_tree,
               'RandomForestClassifier' : model_random_forest,
               'AdaBoostClassifier' : model_adaboost,
               'ExtraTreesClassifier' : model_extra_trees,
               'BaggingClassifier' : model_bagging,
               'XGBClassifier' : model_xgboost,
               'CatBoostClassifier' : model_catboost
               }
"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":
    """
       This is just an example, plese change as necceary. Just maintain final output format with proper names of the models as described above.
    """
    X_train = sys.argv[1]
    y_train = sys.argv[2]
    X_test = sys.argv[3]
    y_test = sys.argv[4]
    
    results = {}
    
    for model_name, func in name2models.items():
        if model_name =='CatBoostClassifier':
            try:
                acc,f1 = func(X_train, y_train, X_test, Y_test)
                results[model_name] = {'acc' : acc, 'f1' : f1}
                print(model_name, acc, f1)
            except:
                continue
        acc,f1 = func(X_train, y_train, X_test, Y_test)
        results[model_name] = {'acc' : acc, 'f1' : f1}
        print(model_name, acc, f1)
  
    

