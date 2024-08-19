import pandas as pd
from DataStream.base import Detector
from scipy import stats


class WRS(Detector):
    
    def __init__(self, threshold:float, window_size:int) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.w1 = None
        self.w2 = None
        self.n_features = None
        
    def __call__(self, current_window:pd.DataFrame) -> None:
        self.w2 = current_window.copy()
    
    def fit(self, X_ref_window: pd.DataFrame, y_ref_window: pd.DataFrame) -> None:
        if self.w1 is None:
            self.w1 = X_ref_window.copy()
            self.w2 = X_ref_window.copy()
            _, self.n_features = X_ref_window.shape
    
    def detect(self, current_window:pd.DataFrame) -> bool:
        for j in range(0, self.n_features):
            _, p_value = stats.ranksums(self.w1.iloc[:,j], self.w2.iloc[:,j])        
            if (p_value <= self.threshold):
                self.w1 = self.w2
                return True
        return False
        