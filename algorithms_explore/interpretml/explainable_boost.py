import os
import time

from interpret import set_visualize_provider
import pandas as pd
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

def read_data(uri="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
              columns= ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender","CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", 
    "Income"]) -> pd.DataFrame:
    df = pd.read_csv(uri,header=None)
    df.columns = columns
    return df

def prepare_train_data(df: pd.DataFrame, sample_frac=0.1,seed=31,
                        test_size: float=0.2):
    temp_df = df.sample(frac=sample_frac)
    X = temp_df.iloc[:,:-1]
    y = temp_df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=seed)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train,seed=45):
    ebm = ExplainableBoostingClassifier(random_state=seed)
    ebm.fit(X_train, y_train)
    return ebm

if __name__=="__main__":
    df = read_data()
    X_train, X_test, y_train, y_test = prepare_train_data(df)
    ebm = train_model(X_train, y_train)
    ebm_global = ebm.explain_global()
    print(ebm_global)
    show(ebm_global)

