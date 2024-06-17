import pandas as pd
from absc.detector import Detector


class IBDD(Detector):
    
    def __init__(self, consecutive_values:float, n_runs:int) -> None:
        self.consecutive_values = consecutive_values
        self.self.n_runs = n_runs
        self.superior_threshold = None
        self.inferior_threshold = None
        self.nrmse = None
        self.w1 = None
        self.w2 = None
        self.files2del = ['w1.jpeg', 'w2.jpeg', 'w1_cv.jpeg', 'w2_cv.jpeg']
        
    def __call__(self, current_window:pd.DataFrame) -> None:
        pass
    
    def fit(self, ref_window:pd.DataFrame) -> None:
        pass
    
    def detect(self, current_window:pd.DataFrame) -> bool:
        pass
        