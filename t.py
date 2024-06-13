import pandas as pd
import numpy as np
from detectors.cdt import CDT
from stream.slidingWindow import SlidingWindow
from sklearn.ensemble import RandomForestClassifier

window_size = 1000

train = pd.read_csv("datasets/train/AedesSex.train.csv")
test = pd.read_csv("datasets/test/AedesSex.test.csv")

train['class'] = train["class"].replace(2, 0)
test["class"] = test["class"].replace(2, 0)

start_window = train.iloc[:window_size]

classifier = RandomForestClassifier(n_estimators=200)

cdt = CDT(classifier, n_samples=1000)
cdt.fit(start_window.iloc[:, :-1])


slinding_window = SlidingWindow(start_window=start_window, stream=test, has_context=True)

for i, window in enumerate(slinding_window):
    print(i, end="\r")
    cdt(window.features())
    if cdt.detect():
        cdt.fit(window.window)
    
    
    