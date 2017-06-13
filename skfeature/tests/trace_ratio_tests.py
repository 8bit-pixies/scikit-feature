from nose.tools import *
import scipy.io
from skfeature.function.similarity_based import trace_ratio
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

def test_trace_ratio():
    # load data
    mat = scipy.io.loadmat('./data/colon.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # reduce cols to speed up test - rather than wait a minute
    X = X[:, :30]  
    num_fea = 5

    # split data into 10 folds
    kfold = KFold(n_splits=10, shuffle=True)
    
    # build pipeline
    pipeline = []
    pipeline.append(('select top k', SelectKBest(score_func=trace_ratio.trace_ratio, k=num_fea)))
    pipeline.append(('linear svm', svm.LinearSVC()))
    model = Pipeline(pipeline)
    
    results = cross_val_score(model, X, y, cv=kfold)
    print("Accuracy: {}".format(results.mean()))
    assert_true(results.mean() > 0.1)
