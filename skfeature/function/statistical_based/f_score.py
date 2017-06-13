import numpy as np
from sklearn.feature_selection import f_classif
from scipy.stats import rankdata

def f_score(X, y, mode="rank"):
    """
    This function implements the anova f_value feature selection (existing method for classification in scikit-learn),
    where f_score = sum((ni/(c-1))*(mean_i - mean)^2)/((1/(n - c))*sum((ni-1)*std_i^2))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y : {numpy array},shape (n_samples,)
        input class labels

    Output
    ------
    F: {numpy array}, shape (n_features,)
        f-score for each feature
    """
    if mode not in ['rank', 'raw', 'index']:
        print('mode is not one of "rank", "raw", "index"')
        raise()
    
    F, pval = f_classif(X, y)
    
    if mode == "raw":
        return F
    elif mode == 'rank':
        return rankdata(F)
    else:
        idx = np.argsort(F)
        return idx[::-1]


