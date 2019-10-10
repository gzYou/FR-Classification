# coding: utf-8

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif,SelectFromModel
from sklearn.linear_model import LogisticRegression

def load_data_from_xlsx(path="./src_data.xlsx",all_features=True):
    """
    Load data from file and decide whether 4 features or all features are kept. 
    """
    data = pd.read_excel(path)
    if not all_features:
        keep_features = ["A_NIH", "A_THALAMUS2", "MIDDLE_BAO", "BATMAN", "FR"]
        data = data.drop(
            [feature for feature in data.columns if feature not in keep_features],
            axis=1,
        )
    print("Data loaded from xlsx.")
    return data



def one_hot_encode(data,categoricals=["A_THALAMUS2", "MIDDLE_BAO", "BATMAN"]):
    """One-hot encode categorical data"""
    for column in categoricals:
        data = pd.get_dummies(data, columns=[column], prefix=[column])
    
    print("One-hot encode categorical data.")
    return data




def scale(X_train):
    """Scale independent variables"""
    X_scaler = preprocessing.StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train.values.astype(float))

    return [X_train_scaled,X_scaler]




def split_pipeline(data, output='FR',test_size=.3,use_scaled=False,use_featureSelection=True,use_LASSO=True):
    """
    Split data into variables
    Arguments: Pandas dataframe, output column (dependent variable),size of test set(account for all data)
    Returns: List of scaled and unscaled dependent and independent variables
    """
    y, X = data.iloc[:, -1], data.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([output], axis=1), data[output], test_size=test_size, random_state=1)
    if use_scaled:
        [X_train,X_scaler] = scale(X_train)
    else:
        X_scaler = None
    if use_featureSelection:
        if use_LASSO:
            selected_columns,dropped_columns = feature_selection_l1(X_train,y_train)
        else:
            selected_columns,dropped_columns = feature_selection_univariate(X_train,y_train,keep=5)
        X_train = X_train.drop(dropped_columns,axis=1)
        X_test = X_test.drop(dropped_columns,axis=1)
    else:
        selected_columns,dropped_columns = "ALL Features",None
    
    enc = preprocessing.OneHotEncoder()
    y_train = enc.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
    y_test = enc.fit_transform(np.array(y_test).reshape(-1,1)).toarray()
    return [np.array(X_train),y_train,np.array(X_test),y_test,X_scaler,selected_columns]



class Recording():
    def __init__(self,index=None): 
        if not index:
            self.index = (
            [
                "training_epochs",
                "batch_size",
                "is_lr_decay",
                "init_learning_rate",
                "decay_rate",
                "decay_steps",
                "num_hidden_layers",
                "num_units_each_layer",
                "is_dropout",
                "dropout_rate",
            ]
            + ["accuracy_" + str(index) for index in range(1, 11)]
            + ["auc_" + str(index) for index in range(1, 11)]
            + ["avg_accuracy", "avg_auc"]
            + ["selected_features_" + str(index) for index in range(1, 11)]
        )
        else:
            self.index = index
    def getIndex(self):
        return self.index
    def getRecordFile(self):
        '''
        Define a dataframe used to record every experiment
        Arguments:parameters to be recored
        '''
        return pd.DataFrame(columns=self.index)
    

def feature_generation(data):
    numerical_features = [
        "LVOD",
        "LVOP",
        "A_MAP",
        "A_NIH",
        "A_GCS",
        "A_WBC",
        "A_GLU",
        "A_SCR",
        "A_PCASPECTS",
        "A_PMI",
        "A_BERN",
        "PCOA",
        "BATMAN"
    ]
    cat_features = list(data.columns.drop(numerical_features+['FR']))
    interactions = pd.DataFrame(index=data.index)
    for col1,col2 in itertools.combinations(cat_features,2):
        new_col_name = col1 + "_" + col2
        new_values = data[col1].map(str) + '_' + data[col2].map(str)
        encoder = preprocessing.LabelEncoder()
        interactions[new_col_name] = encoder.fit_transform(new_values)
    return data.join(interactions)

def feature_selection_univariate(Xtrain,ytrain,keep=5):
    selector = SelectKBest(mutual_info_classif,k=keep)
    Xtrain_new = selector.fit_transform(Xtrain,ytrain)
    selected_features = pd.DataFrame(
        selector.inverse_transform(Xtrain_new),
        index = Xtrain.index,
        columns = Xtrain.columns
    )
    selected_columns = selected_features.columns[selected_features.var() != 0]
    dropped_columns = selected_features.columns[selected_features.var() == 0]
    
    return selected_columns,dropped_columns

def feature_selection_l1(Xtrain,ytrain,c=0.07):
    """ Return selected features using logistic regression with an L1 penalty """
    logistic = LogisticRegression(C=c, penalty="l1", random_state=7).fit(Xtrain,ytrain)
    model = SelectFromModel(logistic,prefit=True)
    Xtrain_new = model.transform(Xtrain)
    selected_features = pd.DataFrame(model.inverse_transform(Xtrain_new),
                                    index = Xtrain.index,
                                    columns = Xtrain.columns)
    
    selected_columns = selected_features.columns[selected_features.var() != 0]
    dropped_columns = selected_features.columns[selected_features.var() == 0]
    
    return selected_columns,dropped_columns
    
def data_pipeline(use_onehotEncoder=False,use_featureGeneration=True,use_featureSelection=True,use_LASSO=True):
    data = load_data_from_xlsx()
    if use_onehotEncoder:
        data = one_hot_encode(data)
    if use_featureGeneration:
        data = feature_generation(data)
    X_train, y_train, X_test, y_test, X_scaler, selected_columns = split_pipeline(data,use_scaled=False,use_featureSelection=use_featureSelection)
    print("Training set and test set are generated.")
    return X_train, y_train, X_test, y_test, X_scaler, selected_columns