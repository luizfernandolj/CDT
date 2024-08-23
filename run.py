import os
import shutil
import time
import pandas as pd
import numpy as np
import json
#from detectors.cdt import CDT
from detectors import *
from DataStream import *
from sklearn.ensemble import RandomForestClassifier
import argparse

def initialize_ibdd_folder(folder:str) -> None:
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def run(dataset:str,
        window_size:int,
        path_train:str,
        path_test:str,
        path_results:str,
        classifier:np.any,
        detector_name:str,
        detector:Detector=None,
        class_index:int=-1,
        context_index:int=None):
    print()
    
    ibdd_dir = f"{os.getcwd()}/detectors/for_ibdd/{dataset}"
    initialize_ibdd_folder(ibdd_dir)
    
    # IMPORTING DATASETS
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    
    # SPLITTING DATASET
    smaller = class_index if class_index < context_index else context_index
    
    # Splitting the train data
    X_train = train.iloc[:, :smaller]
    y_train = train.iloc[:, class_index]
    
    
    
    # ============================== SPLIT THE STREAM DATA =================================
    X_test = test.iloc[:, :smaller]
    y_test = test.iloc[:, class_index]
    
    
    if context_index is not None:
        context_train = train.iloc[:, context_index]
        context_test = test.iloc[:, context_index]

    REF_WINDOW = Window(X_train.iloc[-window_size:], y_train.iloc[-window_size:], context_train)


    STREAM = SlidingWindow(REF_WINDOW, X_test, y_test, context_test, window_size=window_size)
    
    # FITTING OBJECTS
    classifier.fit(X_train, y_train)
    
    detector.fit(X_train, y_train)
    
    
    
    result = {"drifs_detected":0, 
              "drifs_detected_at":[], 
              "time (s)":0, 
              "context_portion":0,
              "classification":np.zeros(len(test))}
    
    proportions = []
    
    start = time.time()
    # RUNNING SLIDING WINDOW
    for i, window in enumerate(STREAM):
        print(f"instance {i+1}/{len(STREAM)}", end="\r")
        
        #print(window.window)
        # Getting the prevalences
        proportions.append(window.get_prevalence(1))
        
        # Predicting the first row of the stream
        classification = classifier.predict(window.X.tail(1))[0]
        
        if classification == window.y.iloc[-1]:
            result["classification"][i] = 1

        detector(window.X)
        
        # Detecting the drift
        detected = detector.detect(window.X)
        
        if detected:
            print(f"drift detected at {i}")
            result["drifs_detected_at"].append(i)
            result["drifs_detected"]= result["drifs_detected"] + 1
            
            if window.get_instances_context(2) is not None and result["context_portion"] == 0:
                context_portion = len(window.get_instances_context(2))/window_size
                result["context_portion"] = context_portion
            
            classifier.fit(window.X, window.y)
            detector.fit(window.X, window.y)
            
            STREAM.switch()
            
    end = time.time()
    
    result["time (s)"] = round(end - start, 3)
    result["classification"] = result["classification"].tolist()
    
    with open(f"{path_results}/{dataset}_{detector_name}.json", "w") as detec_inf:
        json.dump(result, detec_inf)
    
    with open(f"{path_results}/{dataset}_proportions.json", "w") as prop:
        json.dump(proportions, prop)
        




if __name__ == '__main__':
    
    files2del = ['w1.jpeg', 'w2.jpeg', 'w1_cv.jpeg', 'w2_cv.jpeg']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('window_size', type=int)
    parser.add_argument('detector', type=str)
    args = parser.parse_args()
    
    
    print(f"Starting -> {args.detector} on {args.dataset}")
    
    
    path_train = f"{os.getcwd()}/datasets/train/{args.dataset}.train.csv"
    path_test = f"{os.getcwd()}/datasets/test/{args.dataset}.test.csv"
    path_results = f"{os.getcwd()}/results"
    
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf_cdt = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    
    detector = None
    
    if args.detector == "CDT":
        p = 1.5
        train_size = 0.8
        detector = CDT(classifier=clf_cdt, p=p, train_split_size=train_size)
    if args.detector == "IKS":
        ca = 1.95
        detector = IKS(ca=ca)
    if args.detector == "IBDD":
        epsilon = 40
        detector = IBDD(consecutive_values=epsilon, n_runs=20, dataset=args.dataset, window_size=args.window_size)
    if args.detector == "WRS":
        threshold = 0.001
        detector = WRS(threshold=threshold, window_size=args.window_size)
    if args.detector == "BASELINE":
        detector = BASELINE() 
        
    
    class_index = -2
    context_index = -1
    
    
    run(args.dataset, 
        args.window_size, 
        path_train, 
        path_test, 
        path_results, 
        classifier, 
        detector.__class__.__name__, 
        detector,
        class_index,
        context_index,)
    
    for f in files2del:
        if os.path.isdir(f"detectors/for_ibdd/{args.dataset}/{f}"):
            os.remove(f"detectors/for_ibdd/{args.dataset}/{f}")
    
    print(f"\nEnd {args.dataset} with {args.detector}")
    
    
    
    