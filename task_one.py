import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


def init_log(level=None):
    '''
    init log with stdout
    '''
    import logging
    import sys

    log = logging.getLogger('sy')
    log.setLevel(level or logging.INFO)
    format = logging.Formatter("%(asctime)s-%(levelname)s - %(message)s")    # output format
    sh = logging.StreamHandler(stream=sys.stdout)    # output to standard output
    sh.setFormatter(format)
    log.handlers = [] # disable old handler, for rerun the code block
    log.addHandler(sh)
    return log


log = logging.getLogger('sy')


def load_data():
    """
    load data, and split data as train data(70%), validate data(30%), test data
    :return:
        'train_data_X': DataFrame, train data
        'val_data_X': DataFrame, validate data
        'train_data_y': DataFrame, class of train data
        'val_data_y': DataFrame, class of train data
        ''test_data': DataFrame, raw data for test
    """

    model_data = pd.read_csv('model_data.csv')
    model_data = model_data.fillna(value=-1)
    train_data = model_data[model_data.dataset == 'train']
    test_data = model_data[model_data.dataset == 'test']

    train_X = train_data.iloc[:, :-3]
    train_y = train_data.iloc[:, -3]
    train_data_X, val_data_X, train_data_y, val_data_y = train_test_split(train_X, train_y, test_size=0.3,
                                                                          random_state=1)
    return train_data_X, val_data_X, train_data_y, val_data_y, test_data


def calculate_auc_for_threshold(model, train_data_X, train_data_y, val_data_X, val_data_y, threshold, model_init_callback):
    """
    calculate auc for special threshold, apply feature selection with this threshold, then build and validate the
    performance based on the new data. return the related auc for train and validate data
    :param model: pre fitted model,  used to select best feature
    :param train_data_X: train data
    :param train_data_y: class of train data
    :param val_data_X: validate data
    :param val_data_y: class of validate data
    :param threshold: threshold
    :param model_init_callback: init model callback, used to init new training model
    :return:
        'train_auc': auc on train data
        'val_auc': auc on validate data
    """

    from sklearn.metrics import roc_auc_score
    # select features using threshold
    selection = SelectFromModel(model, threshold=threshold, prefit=True)
    select_X_train = selection.transform(train_data_X)
    # train model
    selection_model = model_init_callback()
    selection_model.fit(select_X_train, train_data_y)
    # roc on train
    train_y_pred_prob = selection_model.predict_proba(select_X_train)[:, 1]
    train_auc = roc_auc_score(train_data_y, train_y_pred_prob)
    # roc on val
    select_X_val = selection.transform(val_data_X)
    val_y_pred_prob = selection_model.predict_proba(select_X_val)[:, 1]
    val_auc = roc_auc_score(val_data_y, val_y_pred_prob)
    return train_auc, val_auc


def get_feature_importances(model, norm_order=1):
    """Retrieve or aggregate feature importances from estimator"""

    importances = getattr(model, "feature_importances_", None)

    if importances is None and hasattr(model, "coef_"):
        if model.coef_.ndim == 1:
            importances = np.abs(model.coef_)

        else:
            importances = np.linalg.norm(model.coef_, axis=0,
                                         ord=norm_order)

    elif importances is None:
        raise ValueError(
            "The underlying estimator %s has no `coef_` or "
            "`feature_importances_` attribute. Either pass a fitted estimator"
            " to SelectFromModel or call fit before calling transform."
            % model.__class__.__name__)

    return importances


def init_XGBClassifier():
    """
    init xgboost classifier
    :return:
    """
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=30, random_state=1)
    return clf


def init_LogisticRegression():
    """
    init logistic regression model
    :return:
    """
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(penalty='l1',
                       solver='liblinear',
                       class_weight='balanced',
                       random_state=1)


def init_RandomForestClassifier():
    """
    init random forest classifier
    :return:
    """

    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=30, random_state=1)


def build_model(train_data_X, val_data_X, train_data_y, val_data_y, test_data, model_init_callback, feature_limit=20):
    """
    the main process for build model.
    step 1: train model based on train data
    step 2: validate different feature combination and select the best one
    step 3: train the final model based on the selected features
    step 4: build and return predict result on test data
    :param train_data_X: train data
    :param train_data_y: class of train data
    :param val_data_X: validate data
    :param val_data_y: class of validate data
    :param threshold: threshold
    :param model_init_callback: init model callback, used to init new training model
    :param feature_limit: limit the max number of feature, default is 20
    :return:
        'pred_result_df': DataFrame, with index is id of raw test data, have only one column('pred'), is the
        predict probability for related record
    """
    log.info('build model')
    clf = model_init_callback()
    clf.fit(train_data_X, train_data_y)

    log.info('feature select')
    thresholds = np.sort(get_feature_importances(clf))[::-1][:feature_limit]
    val_aucs = []
    for i, thresh in enumerate(thresholds, 1):
        train_auc, val_auc = calculate_auc_for_threshold(clf, train_data_X, train_data_y, val_data_X, val_data_y, thresh, model_init_callback)
        val_aucs.append(val_auc)
        log.debug('auc for top %d features, score on train is:%f, score on val is: %f', i, train_auc, val_auc)

    # find the best feature threshold
    val_auc_df = pd.DataFrame({'auc': val_aucs})
    index = val_auc_df.rolling(3, min_periods=1).mean()['auc'].values.argmax()
    best_threshold = thresholds[index]
    log.debug('find best threshold value: %f, feature numbers: %d', best_threshold, index)
    # select features using threshold
    selection = SelectFromModel(clf, threshold=best_threshold, prefit=True)
    final_test_data_X = test_data.iloc[:, :-3].reindex(index=test_data['id'])
    select_X_train = selection.transform(train_data_X)
    select_X_test = selection.transform(final_test_data_X)

    # train model
    selection_model = model_init_callback()
    selection_model.fit(select_X_train, train_data_y)
    # roc on train
    test_y_pred_prob = selection_model.predict_proba(select_X_test)[:, 1]
    pred_result_df = pd.DataFrame({'pred': test_y_pred_prob}, index=final_test_data_X.index)
    log.info(pred_result_df)
    return pred_result_df

if __name__ == '__main__':

    init_log(logging.DEBUG)
    log.info('load data')
    train_data_X, val_data_X, train_data_y, val_data_y, test_data = load_data()

    log.info('strategy for xgboost')
    xgboost_pred_result_df = build_model(train_data_X, val_data_X, train_data_y, val_data_y, test_data, init_XGBClassifier, 60)
    log.info('strategy for lr')
    lr_pred_result_df = build_model(train_data_X, val_data_X, train_data_y, val_data_y, test_data, init_LogisticRegression, 100)
    log.info('strategy for rf')
    rf_pred_result_df = build_model(train_data_X, val_data_X, train_data_y, val_data_y, test_data, init_RandomForestClassifier, 60)

    final_result_df = pd.DataFrame({'lr': lr_pred_result_df['pred']
                                    ,'xgboost': xgboost_pred_result_df['pred']
                                    , 'rf': rf_pred_result_df['pred']},
                                   index=xgboost_pred_result_df.index,
                                   columns=['lr', 'xgboost', 'rf'])
    final_result_df.index.name = 'id'
    log.info('final result: %s', final_result_df.to_string())
    final_result_df.to_csv('prediction.csv', sep='\t')
