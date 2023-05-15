import pandas as pd
import numpy as np
import inspect
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class AdaBoost:
    
    def __init__(self, alpha_type):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.alpha_type = alpha_type

    def fit(self, X, y, M=100):
        #X: independent variables - array-like matrix
        #y: target variable - array-like vector
        #M: number of boosting rounds. Default is 100 - integer

        self.alphas = []
        self.G_M = []
        self.training_errors = []
        self.M = M

        for m in range(0, M):
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                w_i = update_weights(w_i, alpha_m, y, y_pred)
            
            G_m = DecisionTreeClassifier(max_depth = 1, max_features= 1)
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            self.G_M.append(G_m)
            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)
            alpha_m = compute_alpha(error_m, y, alpha_type=self.alpha_type)
            self.alphas.append(alpha_m)
        assert len(self.G_M) == len(self.alphas)

    def predict(self, X):
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m
        y_pred = (np.sign(weak_preds.T.sum())).astype(int)
        return y_pred
    
    """ def score(self, X, y):
        ab = AdaBoost()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1111)
        ab.fit(x_train, y_train)
        y_pred = ab.predict(x_test)
        return accuracy_score(y_test, y_pred)
    
    def get_params(self, deep=False):
        d = {'alpha_c': self.alpha_calc}
        return d """
    
# AUXILIAR FUNCTIONS TO THE ADABOOST CLASS

def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error, y_true, alpha_type=0):
    # original
    if alpha_type == 0:
        return 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))
    # new ones
    elif alpha_type == 1:
        unique = y_true.value_counts()
        ratio = min(unique) / sum(unique)
        return ratio * np.log((1 - error + 1e-10) / (error + 1e-10))
    elif alpha_type == 2:
        return error   

def update_weights(w_i, alpha, y, y_pred):
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))