import pandas as pd
import numpy as np
import inspect
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class AdaBoost:
    
    def __init__(self, alpha_type=0):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.alpha_type = alpha_type

    def fit(self, X, y, M=15):
        #X: independent variables - array-like matrix
        #y: target variable - array-like vector
        #M: number of boosting rounds. Default is 100 - integer

        self.alphas = []
        self.G_M = []
        self.training_errors = []
        self.M = M

        #percentage = 0.3  # Example percentage, adjust as needed
        #desired_num_examples = int(len(X) * percentage)

        for m in range(0, M):
            #print("Tamanho =", len(y))
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                w_i = update_weights(w_i, alpha_m, y, y_pred)
                
                # to duplicate the examples in X whose prediction doesnt match the value in y
                mask = (y_pred != y)
                X_mismatched = X[mask]
                y_mismatched = y[mask]
                w_i_mismatched = w_i[mask]
                X = pd.concat([X, X_mismatched], axis=0)
                y = pd.concat([y, y_mismatched], axis=0)
                w_i = np.concatenate((w_i, w_i_mismatched))
            
            G_m = DecisionTreeClassifier(max_depth = 1, max_features= 1)
            G_m.fit(X, y, sample_weight = w_i)

            #X_subset = X.sample(n=desired_num_examples, replace=False)
            #y_subset = y[X_subset.index]
            #valid_indices = np.intersect1d(X.index, X_subset.index)
            #valid_indices = valid_indices[valid_indices < len(w_i)]
            #w_i_subset = w_i[valid_indices]

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
    elif alpha_type == 3:
        return np.exp(error**4)


def update_weights(w_i, alpha, y, y_pred):
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))