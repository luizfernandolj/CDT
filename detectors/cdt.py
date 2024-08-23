import mlquantify as mq
from sklearn.model_selection import train_test_split
from DataStream.base import Detector
import pandas as pd
import numpy as np


class CDT(Detector):
    def __init__(self, classifier, train_split_size: float = 0.5, n_train_test_samples: int = 100, p: int = 3) -> None:
        self.train_split_size = train_split_size
        self.distances = []
        self.threshold = None
        self.p = p
        self.ref_window = None
        self.n_train_test_samples = n_train_test_samples
        self.dys = mq.methods.DyS(classifier)
      
        
    def __call__(self, current_window):
        pass
    
    
    def _get_threshold(self, X, Y):

        args = [(prev, X, Y, self.dys) for prev in np.linspace(0, 1, self.n_train_test_samples)]
        
        train_distances = np.asarray(
            mq.utils.parallel(
                make_artificiall_sample,
                args,
                n_jobs=-1
            )
        )
        train_distances = np.append(train_distances, self.dys.best_distance(X))
        
        mean = np.mean(train_distances)
        std = np.std(train_distances)
    
        return mean + (self.p*std)
    


    def _create_train_test(self, X_window: pd.DataFrame, y_window:pd.DataFrame) -> tuple:
        X_train, X_val, y_train, y_val = train_test_split(X_window, 
                                                          y_window, 
                                                          train_size=self.train_split_size, 
                                                          shuffle=False,
                                                          random_state=32)
        
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
                
        return X_train, y_train, X_val, y_val


    def fit(self, X_ref_window: pd.DataFrame, y_ref_window: pd.DataFrame) -> None:
        
        X, Y, X_val, Y_val = self._create_train_test(X_ref_window, y_ref_window)
        
        self.dys.fit(X, Y)
        
        self.threshold = self._get_threshold(X_val, Y_val)
        
        return self

    def detect(self, X_current_window) -> bool:
        distance = self.dys.best_distance(X_current_window)
        return distance >= self.threshold


def make_artificiall_sample(args):
    pos, X, y, dys = args
    
    prev = [1-pos, pos]
    indexes = mq.utils.generate_artificial_indexes(y, prev, len(X), np.unique(y))
    
    X_sample = X.iloc[indexes]
    distance = dys.best_distance(X_sample)
    return distance