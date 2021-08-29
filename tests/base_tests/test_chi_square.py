import unittest

import scipy.io
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from skfeature.function.statistical_based import chi_square


@unittest.skip("temporarily disabled")
def test_chi2():
    # load data
    import os

    print(os.getcwd())
    mat = scipy.io.loadmat("./data/BASEHOCK.mat")
    X = mat["X"]  # data
    X = X.astype(float)
    y = mat["Y"]  # label
    y = y[:, 0]
    n_samples, n_features = X.shape  # number of samples and number of features

    # perform evaluation on classification task
    num_fea = 100  # number of selected features

    # build pipeline
    pipeline = []
    pipeline.append(("select top k", SelectKBest(score_func=chi_square.chi_square, k=num_fea)))
    pipeline.append(("linear svm", svm.LinearSVC()))
    model = Pipeline(pipeline)

    # split data into 10 folds
    kfold = KFold(n_splits=2, shuffle=True)

    results = cross_val_score(model, X, y, cv=kfold)
    print("Accuracy: {}".format(results.mean()))
    assert results.mean() > 0.95
