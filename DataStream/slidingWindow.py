import pandas as pd
import numpy as np
from copy import deepcopy
from .base import Window

class SlidingWindow:
    """SlidingWindow class for creating a sliding window over a data stream with features, labels, and optional context."""
    
    def __init__(self, reference_window : Window, X, y, context=None, window_size:int=1000) -> None:
        self.window_size = window_size
        self.index = 0
        self.context = context
        
        # Create the initial window (reference window) and stream
        self.reference_window = reference_window
        self.stream = self._create_stream(X, y, context)
        self.actual_window = deepcopy(self.reference_window)
    
    def _create_stream(self, X, y, context):
        """Creates the complete data stream."""
        
        if context is not None:
            stream = pd.concat([X, y, context], axis=1).reset_index(drop=True)
        else:
            stream = pd.concat([X, y], axis=1).reset_index(drop=True)
            
        return stream
    
    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        if self.index >= len(self.stream):
            raise StopIteration
        # Slide the window
        if self.context is not None:
            X_row = self.stream.iloc[self.index, :-2]
            y_row = self.stream.iloc[self.index, -2]
            context_row = self.stream.iloc[self.index, -1]
        else:
            X_row = self.stream.iloc[self.index, :-1]
            y_row = self.stream.iloc[self.index, -1]
            context_row = None
        
        self.actual_window.append(X_row, pd.Series(y_row), pd.Series(context_row))
        
        self.index += 1
        
        return self.actual_window
    
    def __call__(self, func, *args):
        return func(self.reference_window, self.actual_window, *args)
    
    def get_actual_instance(self):
        return self.stream.iloc[self.index]
    
    def get_actual_context(self):
        if self.context is not None:
            return self.get_actual_instance().iloc[-1]
        raise KeyError("Context not available.")
    
    def switch(self):
        self.reference_window = deepcopy(self.actual_window)
        
    def __len__(self):
        return len(self.stream)
