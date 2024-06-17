import pandas as pd
import numpy as np

class SlidingWindow():
    
    def __init__(self, start_window: object, stream:object, has_context=False) -> None:
        self.start_window = start_window.reset_index(drop=True)
        self.actual_window = start_window.reset_index(drop=True)
        self.all_stream = stream
        self.has_context=has_context
        self.window_size = len(start_window)
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.all_stream):
            raise StopIteration
        
        self.actual_window = pd.concat([self.actual_window, self.all_stream.loc[[self.index]]], axis=0).iloc[1:].reset_index(drop=True)
        self.index += 1
        
        return Window(self.actual_window, self.has_context)
    
    def __call__(self, func, *args):
        return func(Window(self.start_window, self.has_context), Window(self.actual_window, self.has_context), *args)
    
    
    def get_actual_instance(self, with_context=False):
        if self.index != 0:
            if self.has_context and not with_context:
                return self.all_stream.loc[[self.index-1]].iloc[:, :-1].reset_index(drop=True)
            else:
                return self.all_stream.loc[[self.index-1]].iloc[:, :].reset_index(drop=True)
            
        return self.all_stream.loc[[self.index]].reset_index(drop=True)
    
    def get_actual_context(self):
        if self.has_context:    
            return self.get_actual_instance(with_context=True).iloc[:, -1].loc[0]
        else:
            raise KeyError("Context not instanciated, please set the context list first")
        
    def set_context(self, context_list: list):
        self.all_stream["context"] = context_list
        self.has_context = True
        
    
    def switch(self):
        self.start_window = self.actual_window
            
        
    

class Window:
    
    def __init__(self, window: object, has_context: bool):
        self.window = window
        self.has_context = has_context
        self.index = 0   
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.window):
            raise StopIteration
        
        row = self.window.iloc[self.index]
        self.index += 1
        return row
        
    
    def features(self):
        c = 2 if self.has_context else 1
        return self.window.iloc[:, :-c].reset_index(drop=True)
    
    def labels(self):
        c = 2 if self.has_context else 1
        return self.window.iloc[:, -c].reset_index(drop=True)
    
    def get_prevalence(self, return_class=1):
        return self.labels().value_counts(normalize=True)[return_class]
    
    def get_instances_context(self, context:int) -> pd.DataFrame:
        if self.has_context:
            context_df = self.window.query("context == @context")
            if context != None:
                return context_df
            else:
                return False

               
    
    def __str__(self):
        return f"{self.window}"