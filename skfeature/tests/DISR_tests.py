from nose.tools import *
import scipy.io
from skfeature.function.information_theoretical_based import DISR
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

def test_disr():
    # load data
    mat = scipy.io.loadmat('./data/colon.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # reduce the sample to speed up the test
    X = X[:, :30]
    # perform evaluation on classification task
    num_fea = 10    # number of selected features

    # build pipeline
    pipeline = []
    pipeline.append(('select top k', SelectKBest(score_func=DISR.disr, k=num_fea)))
    pipeline.append(('linear svm', svm.LinearSVC()))
    model = Pipeline(pipeline)

    # split data into 10 folds
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(model, X, y, cv=kfold)
    print("Accuracy: {}".format(results.mean()))
    assert_true(results.mean() > 0.1)

if __name__ == '__main__':
    main()
