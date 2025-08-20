import lightgbm as lgb
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
from smolagents import tool

@tool
def run_lightgbm(feature_fraction: float) -> dict:
    """
    Train and evaluate a LightGBM binary classifier on the Breast Cancer dataset.

    This function loads the Breast Cancer dataset from scikit-learn,
    splits it into training and validation sets, trains a LightGBM model
    with the specified feature_fraction parameter, applies early stopping,
    and evaluates prediction accuracy on the validation set.
    Example:
        >>> run_lightgbm(0.8)
        Prediction accuracy 0.96

    Args:
        feature_fraction (float): Fraction of features to use when training
            each boosting iteration. Must be between 0.0 and 1.0.
    Return:
        dict: a dictionary with the accuracy
    """
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)

    # Train/validation split
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        data, target, test_size=0.25, random_state=42
    )

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "feature_fraction": feature_fraction,
    }

    # Prepare LightGBM datasets
    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)

    # Train with early stopping
    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Predict and evaluate accuracy
    preds = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
    pred_labels = (preds >= 0.5).astype(int)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    print("Prediction accuracy", accuracy)
    return  {"Accuracy": accuracy}

