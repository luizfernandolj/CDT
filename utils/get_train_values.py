from utils.getTrainingScores import getTrainingScores
from utils.getCalibratedTrainingScores import getCalibratedTrainingScores



def get_train_values(X, Y, n_folds, classifier):
    scores, classifier = getCalibratedTrainingScores(X, Y, n_folds, classifier)
    pos_scores = scores[scores["class"]==1]["scores"]
    neg_scores = scores[scores["class"]==0]["scores"]

    return pos_scores, neg_scores, classifier