import pandas as pd
import numpy as np
from detectors.cdt import CDT
from stream.slidingWindow import SlidingWindow, Window
from sklearn.ensemble import RandomForestClassifier

window_size = 300

train = pd.read_csv("datasets/train/AedesSex.train.csv")
test = pd.read_csv("datasets/test/AedesSex.test.csv")

train['class'] = train["class"].replace(2, 0)
test["class"] = test["class"].replace(2, 0)

start_window = train.iloc[:window_size]

classifier = RandomForestClassifier(n_estimators=200)

X_train = start_window.drop(["class", "context"], axis=1)
y_train = start_window["class"]

classifier.fit(X_train, y_train)

cdt = CDT(classifier, n_train_test_samples=100)
cdt.fit(start_window)

slinding_window = SlidingWindow(start_window=start_window, stream=test, has_context=True)

def detec(ref_window: Window, current_window: Window, detector) -> bool:
    return detector.detect(current_window.features())


for i, window in enumerate(slinding_window):
    print(i, end="\r")
    detected = slinding_window(detec, cdt)
    print(slinding_window.get_actual_context())
    if detected:
        print(detected)
        classifier.fit(window.features(), window.labels())
        cdt.fit(window.window)
    
    
    
    