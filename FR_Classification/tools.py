# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


def load_data_from_xlsx(path="./src_data.xlsx",all_features=False):
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




def split_pipeline(data, output='FR',test_size=.2):
    """
    Split data into variables
    Arguments: Pandas dataframe, output column (dependent variable),size of test set(account for all data)
    Returns: List of scaled and unscaled dependent and independent variables
    """
    y, X = data.iloc[:, -1], data.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([output], axis=1), data[output], test_size=test_size, random_state=1)
    [X_train_scaled,X_scaler] = scale(X_train)
    
    enc = preprocessing.OneHotEncoder()
    y_train = enc.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
    y_test = enc.fit_transform(np.array(y_test).reshape(-1,1)).toarray()
    return [np.array(X_train_scaled),y_train,np.array(X_test),y_test,X_scaler]



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
    

def data_pipeline():
    data = load_data_from_xlsx()
    data = one_hot_encode(data)
    X_train_scaled, y_train, X_test, y_test, X_scaler = split_pipeline(data)
    print("Training set and test set are generated.")
    return X_train_scaled, y_train, X_test, y_test, X_scaler