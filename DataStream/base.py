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
    
    @property
    def classifier(self):
        return self.classifier_

    @classifier.setter
    def classifier(self, value):
        self.classifier_ = value
    
    
    

class Window:
    """Window class for just creating a window with features, labels and context if passed
    """
    
    def __init__(self, X, y, context=None):
        self.X = X
        self.y = y
        self.context = context
        self.window = self._get_window(X, y, context)
        self.index = 0
        
    
    def _get_window(self, X, y, context):
        if context is not None:
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
    
    def append(self, X_row, y_row, context_row): 
        # Adiciona a linha `X_row` ao DataFrame `self.X`
        self.X = pd.concat([self.X, X_row.to_frame().T], axis=0, ignore_index=True)[1:]

        # Adiciona a linha `y_row` ao DataFrame `self.y`
        self.y = pd.concat([self.y, y_row], axis=0, ignore_index=True)[1:]

        if self.context is not None:
            # Adiciona a linha `context_row` ao DataFrame `self.context`
            self.context = pd.concat([self.context, context_row], axis=0, ignore_index=True)[1:]
            
            # Concatena `self.X`, `self.y`, e `self.context` em `self.window`
            self.window = pd.concat([self.X, self.y, self.context], axis=1, ignore_index=True)
        else:
            # Concatena apenas `self.X` e `self.y` em `self.window`
            self.window = pd.concat([self.X, self.y], axis=1, ignore_index=True)


    def get_prevalence(self, return_class=None):
        prevs = self.y.value_counts(normalize=True)
        if return_class:    
            return prevs[return_class]
        return prevs
    
    def get_instances_context(self, context:int) -> pd.DataFrame:
        if self.context is not None:
            context_df = self.window[self.window.iloc[:, -1] == context]
            if context_df is not None:
                return context_df
            return None
        raise ValueError("No context was specified")
    
    def __str__(self):
        return f"{self.window}"
    
    