import mlquantify as mq
from DataStream import *

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


DATASET = "AedesQuinx"
WINDOW_SIZE = 1000

train = pd.read_csv(f"datasets/train/{DATASET}.train.csv")
test = pd.read_csv(f"datasets/test/{DATASET}.test.csv")


X_train = train.iloc[:, :-2]
y_train = train.iloc[:, -2]
context_train = train.iloc[:, -1]

dyssyn = mq.methods.DyS(RandomForestClassifier())

dyssyn.fit(X_train.iloc[:WINDOW_SIZE], y_train.iloc[:WINDOW_SIZE])


X_test = test.iloc[:, :-2]
y_test = test.iloc[:, -2]
context_test = test.iloc[:, -1]


X_train_sized = X_train[-WINDOW_SIZE:]
y_train_sized = y_train[-WINDOW_SIZE:]
context_train_sized = context_train[-WINDOW_SIZE:]


REF_WINDOW = Window(X_train_sized, y_train_sized, context_train_sized)


STREAM = SlidingWindow(REF_WINDOW, X_test, y_test, context_test, window_size=WINDOW_SIZE)


distances = []
for i, window in enumerate(STREAM):
    distance = dyssyn.best_distance(window.X)
    print(f"window -> {i+1}/{len(STREAM)} with distance -> {distance}", end="\r")
    
    distances.append(distance)



sns.lineplot(distances)
plt.show()