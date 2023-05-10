import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    
    def __init__(self):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, y, alpha_type=0, M=100):
        #X: independent variables - array-like matrix
        #y: target variable - array-like vector
        #M: number of boosting rounds. Default is 100 - integer

        self.alphas = [] 
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
            alpha_m = compute_alpha(error_m, alpha_type)
            self.alphas.append(alpha_m)
        assert len(self.G_M) == len(self.alphas)

    def predict(self, X):
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)
        return y_pred
    
# AUXILIAR FUNCTIONS TO THE ADABOOST CLASS

def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error, alpha_type=0):
    # original
    if alpha_type == 0:
        return np.log((1 - error) / error)
    # new ones
    if alpha_type == 1:
        return np.log(error + 1e-10)
    if alpha_type == 2:
        return error
    if alpha_type == 3:
        return np.sqrt(error)

def update_weights(w_i, alpha, y, y_pred):
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))