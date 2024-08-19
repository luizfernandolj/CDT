import os
import pandas as pd
from DataStream.base import Detector
from skimage.io import imread
from skimage.metrics import mean_squared_error, structural_similarity
from scipy import stats
from random import seed, shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class IBDD(Detector):
    
    def __init__(self, consecutive_values:float, n_runs:int, dataset:str, window_size:int) -> None:
        self.window_size = window_size
        self.dataset = dataset
        self.consecutive_values = consecutive_values
        self.n_runs = n_runs
        self.threshold_diffs = None
        self.superior_threshold = None
        self.inferior_threshold = None
        self.nrmse = None
        self.w1 = None
        self.w2 = None
        self.last_update = 0
        self.i = 0
        
    def _get_imgdistribution(self, file_name:str, data) -> np.any:
        plt.imsave(f"{os.getcwd()}/detectors/for_ibdd/{self.dataset}/{file_name}", data.transpose(), cmap = 'Greys', dpi=100)
        w = imread(f"{os.getcwd()}/detectors/for_ibdd/{self.dataset}/{file_name}")
        return w
        
    
    def _find_initial_threshold(self, X_train, window_length, n_runs):
        
        w1 = X_train.iloc[-window_length:].copy()
        w1_cv = self._get_imgdistribution(f"w1_cv.jpeg", w1)

        max_index = X_train.shape[0]
        sequence = [i for i in range(max_index)]
        nrmse_cv = []
        for i in range(0,n_runs):
            # seed random number generator
            seed(i)
            # randomly shuffle the sequence
            shuffle(sequence)
            w2 = X_train.iloc[sequence[:window_length]].copy()
            w2.reset_index(drop=True, inplace=True)
            w2_cv = self._get_imgdistribution(f"w2_cv.jpeg", w2)
            nrmse_cv.append(mean_squared_error(w1_cv,w2_cv))
            threshold1 = np.mean(nrmse_cv)+2*np.std(nrmse_cv)
            threshold2 = np.mean(nrmse_cv)-2*np.std(nrmse_cv)
        if threshold2 < 0:
            threshold2 = 0		
        return (threshold1, threshold2, nrmse_cv) 
    
        
    def __call__(self, current_window:pd.DataFrame) -> None:
        self.w2 = self._get_imgdistribution("w2.jpeg", current_window)
        self.nrmse.append(mean_squared_error(self.w1, self.w2))
        if (self.i - self.last_update > 60):
            self.superior_threshold = np.mean(self.nrmse[-50:]) + 2 * np.std(self.nrmse[-50:])
            self.inferior_threshold = np.mean(self.nrmse[-50:]) - 2 * np.std(self.nrmse[-50:])
            self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)
            self.last_update = self.i
    
    def fit(self, X_ref_window: pd.DataFrame, y_ref_window: pd.DataFrame) -> None:
        train_features = X_ref_window
        if self.nrmse is None:
            self.superior_threshold, self.inferior_threshold, self.nrmse = self._find_initial_threshold(train_features,
                                                                                                        self.window_size,
                                                                                                        self.n_runs)
            self.threshold_diffs = [self.superior_threshold - self.inferior_threshold]   
            self.w1 = self._get_imgdistribution("w1.jpeg", train_features[-self.window_size:])
    
    def detect(self, current_window:pd.DataFrame) -> bool:
        if (all(i >= self.superior_threshold for i in self.nrmse[-self.consecutive_values:])):
            self.superior_threshold = self.nrmse[-1] + np.std(self.nrmse[-50:-1])
            self.inferior_threshold = self.nrmse[-1] - np.mean(self.threshold_diffs)
            self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)
            self.last_update = self.i
            return True

        if (all(i <= self.inferior_threshold for i in self.nrmse[-self.consecutive_values:])):
            self.inferior_threshold = self.nrmse[-1] - np.std(self.nrmse[-50:-1])
            self.superior_threshold = self.nrmse[-1] + np.mean(self.threshold_diffs)
            self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)
            self.last_update = self.i
            return True
        self.i = self.i + 1
        return False
        