import numpy as np
from sklearn import datasets, svm
from sklearn.pipeline import Pipeline

from skfeature.function.streaming.alpha_investing import AlphaInvesting

iris = datasets.load_iris()
data, y = iris.data, iris.target


def test_alpha_investing():
    sel = AlphaInvesting(w0=0.05, dw=0.05).fit(data, y)
    assert np.any(sel.get_support())
