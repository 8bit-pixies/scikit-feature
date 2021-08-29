from sklearn.feature_selection import VarianceThreshold


def low_variance_feature_selection(X=None, threshold=0.0, mode="rank"):
    """
    This function implements the low_variance feature selection (existing method in scikit-learn)

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    p:{float}
        parameter used to calculate the threshold(threshold = p*(1-p))

    Output
    ------
    X_new: {numpy array}, shape (n_samples, n_selected_features)
        data with selected features
    """
    if mode == "rank":
        return VarianceThreshold(threshold=threshold)
    else:
        sel = VarianceThreshold(threshold)
        return sel.fit_transform(X)
