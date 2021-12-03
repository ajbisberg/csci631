import diffprivlib as dp
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
np.random.seed(31415)
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import math
import fairlearn.metrics as mf
from multiprocessing import Pool

def get_classifier(cls_str, diffpriv=True, bounds=None, epsilon=None, acc=None):
    '''

    :param cls_str: one of ['NB', 'LR', 'RF']
    :param diffpriv:Whether dp classifier or regular classifier
    :param bounds:  Bounds of the data, provided as a tuple of the form (min, max).
                    min and max can either be scalars, covering the min/max of the
                    entire data, or vectors with one entry per feature. If not
                    provided, the bounds are computed on the data when .fit() is
                    first called, resulting in a PrivacyLeakWarning
    :param epsilon: Privacy parameter for the model. default: 1.0
    :param acc:     Accountant to keep track of privacy budget
    :return:        The classifier instantiated
    '''

    if(cls_str == 'NB' and diffpriv == True):
        return dp.models.GaussianNB(epsilon=epsilon, bounds=bounds, accountant=acc)
    
    elif(cls_str == 'NB' and diffpriv == False):
        return GaussianNB()
    
    elif(cls_str == 'LR' and diffpriv == True):
        return dp.models.LogisticRegression(epsilon=epsilon, bounds=bounds, accountant=acc)
    
    elif(cls_str == 'LR' and diffpriv == False):
        return LogisticRegression(random_state=0)
    
    elif(cls_str == 'RF' and diffpriv == True):
        return dp.models.RandomForestClassifier(epsilon=epsilon, bounds=bounds, accountant=acc)
    
    elif(cls_str == 'RF' and diffpriv == False):
        return RandomForestClassifier(random_state=0)
    
    else:
        print("Incorrect Classifier - must be one of ['NB', 'LR', 'RF']")
        return
    
def train_score_clf(i, eps, cls_str, gender_attr_str, race_attr_str, X_train, y_train, X_test, y_test):
    print('training trial={}, eps={}'.format(i,eps))
    clf = get_classifier(cls_str, diffpriv=True, epsilon=eps)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_train)            
    # fairness metrics need the sensitive column to be binary {0,1}. Cannot normalize!   
    return(
        clf.score(X_test, y_test), # acc
        mf.demographic_parity_ratio(
            y_true=np.array(y_train), y_pred=y_pred, sensitive_features=np.array(X_train[gender_attr_str])),
        mf.demographic_parity_ratio(
            y_true=np.array(y_train), y_pred=y_pred, sensitive_features=np.array(X_train[race_attr_str])),
        mf.equalized_odds_ratio(
            y_true=np.array(y_train), y_pred=y_pred, sensitive_features=np.array(X_train[gender_attr_str])),
        mf.equalized_odds_ratio(
            y_true=np.array(y_train), y_pred=y_pred, sensitive_features=np.array(X_train[race_attr_str])),
    )