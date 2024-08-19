from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any

class Detector(ABC):
    """Abstract class for drift detectors"""
    
    @abstractmethod
    def fit(self, X_ref_window:pd.DataFrame, y_ref_window:pd.DataFrame) -> None:
        pass
    
    @abstractmethod
    def detect(self, current_window:pd.DataFrame) -> bool:
        pass
    
    
    

class Window:
    """Window class for just creating a window with features, labels and context if passed
    """
    
    def __init__(self, X, y, context=None):
        assert isinstance(context, pd.Series) or isinstance(context, np.ndarray) or not context
        assert isinstance(X, pd.Series) or isinstance(X, np.ndarray)
        assert isinstance(Y, pd.Series) or isinstance(Y, np.ndarray) 
        
        self.X = X
        self.y = y
        self.context = context
        self.window = self._get_window(X, y, context)
        self.index = 0
        
    
    def _get_window(X, y, context):
        if context:
            return pd.concat([X, y, context], axis=1, ignore_index=True)
        return pd.concat([X, y], ignore_index=True)    
    
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.X):
            raise StopIteration
        
        row = self.window.iloc[self.index]
            
        self.index += 1
        return row


    def get_prevalence(self, return_class=None):
        prevs = self.y.value_counts(normalize=True)
        if return_class:    
            return prevs[return_class]
        return prevs
    
    def get_instances_context(self, context:int) -> pd.DataFrame:
        if self.context:
            context_df =  self.window[self.window.iloc[:, -1] == context]
            if context_df != None:
                return context_df
            print(f"There is no context {context} in this window")
            return False
        raise ValueError("No context was specified")
    
    def __str__(self):
        return f"{self.window}"
    
    