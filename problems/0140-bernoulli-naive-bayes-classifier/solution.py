import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        n = len(y)
        self.priors = []
        self.p = []
        for c in self.classes:
            c_in_y = np.sum([y == c])
            self.priors.append(c_in_y/n)
            nb_features_in_c = np.sum(X[y == c], axis=0)
            p_feat = (nb_features_in_c + self.smoothing)/(c_in_y + 2 * self.smoothing)
            self.p.append(p_feat)
        self.p = np.array(self.p)
    def predict(self, X):
        # Predict class labels for test set X
        score = X@np.log(self.p).T + (1 - X)@np.log(1 - self.p).T + np.log(self.priors)
        
        return self.classes[np.argmax(score, axis=1)]
        