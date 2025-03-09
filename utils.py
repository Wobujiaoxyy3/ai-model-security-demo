import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):

    data_path = file_path
    df = pd.read_csv(data_path)
    df.drop(columns=["newbalanceOrig", "newbalanceDest", "nameOrig", "nameDest", "isFlaggedFraud"], inplace=True)

    df.fillna(0, inplace=True)

    df["step"] = (df["step"] - df["step"].min()) / (df["step"].max() - df["step"].min())

    df = pd.get_dummies(df, columns=["type"], drop_first=True)
    df = df.astype({col: np.float32 for col in df.select_dtypes(include=['bool']).columns})


    numeric_cols = ["amount", "oldbalanceOrg", "oldbalanceDest", "step"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
