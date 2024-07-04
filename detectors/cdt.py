from absc.detector import Detector
from utils.generate_samples import generate_samples_binary
from quantification.dys_method import get_dys_distance
from utils.get_train_values import get_train_values
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class CDT(Detector):
    def __init__(self, classifier, train_split_size: float = 0.5, n_train_test_samples: int = 100, p: int = 5) -> None:
        self.train_split_size = train_split_size
        self.distances = []
        self.threshold = None
        self.p = p
        self.ref_window = None
        self.classifier = classifier
        self.n_train_test_samples = n_train_test_samples
        self.pos_scores = None
        self.neg_scores = None
        self.scores = None
        
    def __call__(self, current_window):
        pass

    def _create_train_test(self, window: pd.DataFrame) -> tuple:
        window = window.reset_index(drop=True)
        size = int(len(window) * self.train_split_size)
        train, test = window.iloc[:size], window.iloc[size:]
        X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
        return X_train, y_train, test

    def fit(self, X_ref_window: pd.DataFrame, y_ref_window: pd.DataFrame) -> None:
        self.ref_window = pd.concat([X_ref_window, y_ref_window], axis=1)
        X, Y, test = self._create_train_test(self.ref_window)
        self.pos_scores, self.neg_scores, self.classifier = get_train_values(X, Y, 10, self.classifier)
        samples = generate_samples_binary(test, self.n_train_test_samples, len(self.ref_window), label_column=test.columns[-1])
        
        for sample in samples:
            test_scores = self.classifier.predict_proba(sample)[:, 1]
            dys_distance = get_dys_distance(self.pos_scores, self.neg_scores, test_scores)
            self.distances.append(dys_distance)
        
        ref_scores = self.classifier.predict_proba(X_ref_window)[:, 1]
        dys_distance = get_dys_distance(self.pos_scores, self.neg_scores, ref_scores)
        self.distances.append(dys_distance)
        
        self.threshold = np.percentile(self.distances, [self.p, 100 - self.p])
        self.scores = ref_scores

    def detect(self, current_window: pd.DataFrame) -> bool:
        new_instance = current_window.tail(1)
        score = self.classifier.predict_proba(new_instance)[:, 1]
        self.scores = np.concatenate((self.scores, score))[1:]
        dys_distance = np.round(get_dys_distance(self.pos_scores, self.neg_scores, self.scores), 5)
        return dys_distance < self.threshold[0] or dys_distance > self.threshold[1]
