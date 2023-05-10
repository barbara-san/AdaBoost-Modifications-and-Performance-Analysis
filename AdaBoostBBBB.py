import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=1000, learning_rate=1.0, random_state=123):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(n_estimators, dtype=np.float64)

    def fit(self, X_train, y_train, n = 1):
        sample_weight = np.ones(len(X_train)) / len(X_train)
        for i in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=1, max_features=1, random_state=self.random_state)
            estimator.fit(X_train, y_train, sample_weight=sample_weight)
            y_pred = estimator.predict(X_train)
            error = np.sum(sample_weight * (y_pred != y_train))
            if n == 1: #original
                alpha = 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))
            if n == 2:
                alpha = (error + 1e-10)
            if n == 3:
                alpha = error * np.log((error + 1e-10))
            if n == 4:
                alpha = np.sqrt(abs(error))
                
            if n == 5:
                alpha = np.log((error + 1e-10))
            if n == 6:
                alpha = np.log((1 - error + 1e-10))
            if n == 7:
                alpha = np.log((error + 1e-10) * (1 - error + 1e-10))
            
            

            dif = np.array(y_pred != y_train)
            for i in range (self.n_estimators):
                sample_weight[i] *= np.exp(-1*alpha*dif[i])
            sample_weight /= np.sum(sample_weight)
            self.estimators_.append(estimator)
            self.estimator_weights_[i] = alpha * self.learning_rate
            self.estimator_errors_[i] = error

    def predict(self, X_test):
        predictions = np.zeros(len(X_test))
        for i in range(self.n_estimators):
            y_pred = self.estimators_[i].predict(X_test)
            predictions += abs(self.estimator_weights_[i]) * y_pred * self.learning_rate

        return np.sign(predictions)
