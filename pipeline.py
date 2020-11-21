#import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#import descartes
#import censusdata
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import datetime

MODELS = {
    'LogisticRegression': LogisticRegression(), 
    'LinearSVC': LinearSVC(), 
    'GaussianNB': GaussianNB()
}

GRID = {
    'LogisticRegression': [{'penalty': x, 'C': y, 'random_state': 0} 
                           for x in ('l2', 'none') \
                           for y in (0.01, 0.1, 1, 10, 100)],
    'GaussianNB': [{'priors': None}],
    'LinearSVC': [{'C': x, 'random_state': 0} \
                  for x in (0.01, 0.1, 1, 10, 100)]
}


def load_data(filename):

    df = pd.read_csv(filename)
    return df

def generate_basic_exploration(df, date_col=None, size=(10,10)):

    #1 identify time range
    if date_col:
        min_date = df[date_col].min()
        max_date = df[date_col].max()

    #2 generate variable distributions - histograms to look for outliers
    df.hist(figsize=size)

    #3 identify basic correlations
    corr_matrix = df.corr().fillna(0)
    png, ax = plt.subplots(figsize=size)
    ax = sns.heatmap(corr_matrix, center=0, vmin= -1, vmax=1,
    cmap=sns.diverging_palette(250, 10, as_cmap=True))
    plt.show()
    if date_col:
        return min_date, max_date

def summarize_by_group(df, group_col, agg_col):


    gb = df.groupby(group_col)[agg_col].count()
    print('For group ', group_col)
    print('Min is: ', gb.min())
    print('Max is: ', gb.max())
    print('Mean is: ', gb.mean())
    print('Count is: ', gb.count())


def train_test(df):

    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, 
                                                           random_state=13)
    print('Train length is ', len(train), ' choo choo')
    print('Test length is ', len(test))
    return train, test


def impute_missing(train, test, cols=[], assassinate_negatives=False):

    if assassinate_negatives:
        for col in cols: 
            train.loc[train[col] < 0] = np.nan
            test.loc[test[col] < 0] = np.nan

    for col in cols:
        med = train[col].median()
        train[col].fillna(med, inplace=True)
        test[col].fillna(med, inplace=True)
    return train, test

def normalize_cols(train, test, cols=[]):

    for col in cols:
        new_col_name = 'Normalized ' + col
        mean, std = train[col].mean(), train[col].std()
        normalized_train = (train[col] - mean) / std
        normalized_test = (test[col] - mean) / std
        train[new_col_name] = normalized_train
        test[new_col_name] = normalized_test

    return train, test

def hot_potato(train_predictors, test_predictors):

    train_potato = pd.get_dummies(train_predictors)
    test_potato = pd.get_dummies(test_predictors)
    hot_train, hot_test = train_potato.align(test_potato, join='left', axis=1)

    return hot_train, hot_test

def slice_and_dice(df, col, bins):

    cut_col_name = 'Diced ' + col
    diced_col = pd.cut(df[col], bins)
    df[cut_col_name] = diced_col

    return df

def build_classifiers(train, test):

# Begin timer 
    start = datetime.datetime.now()

    # Initialize results data frame 
    results = pd.DataFrame(columns=['Model', 'Params', 'Accuracy'])

    # Loop over models
    index = 0
    for model_key in MODELS.keys():
        
        # Loop over parameters 
        for params in GRID[model_key]: 
            print("Training model:", model_key, "|", params)
            
            # Create model 
            model = MODELS[model_key]
            model.set_params(**params)
            
            # Fit model on training set 
            model.fit(train.iloc[:,:-1], train.iloc[:,-1])
            
            # Predict on testing set 
            predict = model.predict(test.iloc[:,:-1])
            
            # Evaluate predictions 
            accuracy = accuracy_score(test.iloc[:,-1], predict)
            
            # Store results in your results data frame 
            results.loc[index] = model_key, params, accuracy
            index += 1
            
    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)
    return results


def evaluate_classifiers(y_test, y_pred):

    '''
    Prints various evaluation metrics
    Inputs:
        y_test
        y_pred
    Returns:
        None
    '''
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred))




