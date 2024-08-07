import os
import shutil
import time
import pandas as pd
import numpy as np
import json
#from detectors.cdt import CDT
from detectors.cdtcopy import CDT_syn
from detectors.iks import IKS
from detectors.ibdd import IBDD
from detectors.wrs import WRS
from detectors.baseline import BASELINE
from absc.detector import Detector
from stream.slidingWindow import SlidingWindow, Window
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
        detector:Detector,
        detector_name:str,
        has_context:bool):
    print()
    
    if detector is None:
        raise Exception("No detector specified")
    
    ibdd_dir = f"{os.getcwd()}/detectors/for_ibdd/{dataset}"
    initialize_ibdd_folder(ibdd_dir)
    
    if has_context:
        a = 2
    else:
        a = 1
    
    # IMPORTING DATASETS
    train = pd.read_csv(path_train, )
    test = pd.read_csv(path_test, )
    train.replace({list(train.columns)[-a]:{2:0}}, inplace=True)
    test.replace({list(train.columns)[-a]:{2:0}}, inplace=True)
    
    # FITTING CLASSIFIER INTO TRAIN DATASET
    X_train = train.iloc[:, :-a]
    y_train = train.iloc[:, -a]
    classifier.fit(X_train.values, y_train.values)
    
    # START WINDOW
    start_window = train.iloc[-window_size:]
    
    
    detector.fit(X_train, y_train)
    
    sliding_window = SlidingWindow(start_window=start_window, stream=test, has_context=has_context)
    
    
    def f(start_window:Window, current_window:Window, detector:Detector) -> bool:
        return detector.detect(current_window.features())
    
    
    result = {"drifs_detected":0, 
              "drifs_detected_at":[], 
              "time (s)":0, 
              "context_portion":0,
              "classification":np.zeros(len(test))}
    
    proportions = []
    
    start = time.time()
    # RUNNING SLIDING WINDOW
    for i, window in enumerate(sliding_window):
        print(f"instance {i+1}", end="\r")
        
        #print(window.window)
        proportions.append(window.get_prevalence(1))
        classification = classifier.predict(window.features().iloc[[-1]].values)
        if classification == window.labels().iloc[-1]:
            result["classification"][i] = 1

        detector(window.features())
        
        detected = sliding_window(f, detector)
        if detected:
            print(f"drift detected at {i}")
            result["drifs_detected_at"].append(i)
            result["drifs_detected"]= result["drifs_detected"] + 1
            
            if window.get_instances_context(2) is not None and result["context_portion"] == 0:
                context_portion = len(window.get_instances_context(2))/window_size
                result["context_portion"] = context_portion
            
            classifier.fit(window.features().values, window.labels().values)
            detector.fit(window.features(), window.labels())
            sliding_window.switch()
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
    
    
    print(f"Starting -> {args.detector}")
    
    
    path_train = f"{os.getcwd()}/datasets/train/{args.dataset}.train.csv"
    path_test = f"{os.getcwd()}/datasets/test/{args.dataset}.test.csv"
    path_results = f"{os.getcwd()}/results"
    
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf_cdt = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    
    detector = None
    
    if args.detector == "CDT":
        p = 2
        detector = CDT_syn(classifier=clf_cdt, p=p)
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
        
    has_context = True       
    
    
    run(args.dataset, 
        args.window_size, 
        path_train, 
        path_test, 
        path_results, 
        classifier, 
        detector, 
        args.detector,
        has_context)
    
    for f in files2del:
        if os.path.isdir(f"detectors/for_ibdd/{args.dataset}/{f}"):
            os.remove(f"detectors/for_ibdd/{args.dataset}/{f}")
    
    print("\nEnd")
    
    
    
    