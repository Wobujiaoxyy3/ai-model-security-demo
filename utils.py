import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(uri):
    """
    Load and preprocess data from MongoDB:
    1. Remove fields: _id, newbalanceOrig, newbalanceDest
    2. **Sort data by step (time)**
    3. **Normalize step**
    4. Replace nameOrig and nameDest with their historical fraud probability
    """

    # 1. Connect to MongoDB
    client = MongoClient(uri)
    db = client["Bank"]
    collection = db["bank_data_filtered"]

    # 2. Read data (excluding _id, newbalanceOrig, newbalanceDest)
    data = list(collection.find({}, {"_id": 0, "newbalanceOrig": 0, "newbalanceDest": 0, "isFlaggedFraud": 0}))
    df = pd.DataFrame(data)

    # 3. Check if required fields exist
    required_columns = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud", "nameOrig", "nameDest"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing fields: {missing_columns}")

    # 4. Handle missing values
    df.fillna(0, inplace=True)

    # 5. **Sort data by step (time)**
    df = df.sort_values(by="step").reset_index(drop=True)

    # 6. **Normalize step**
    df["step"] = (df["step"] - df["step"].min()) / (df["step"].max() - df["step"].min())

    # 7. One-hot encode transaction type
    # df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # 8. Normalize numeric columns
    numeric_cols = ["amount", "oldbalanceOrg", "oldbalanceDest", "step"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 9. Compute fraud probability for nameOrig and nameDest
    fraud_counts = {}  # Store fraud transaction counts for each account
    total_counts = {}  # Store total transaction counts for each account
    name_orig_probs = []
    name_dest_probs = []

    for _, row in df.iterrows():
        orig, dest, is_fraud = row["nameOrig"], row["nameDest"], row["isFraud"]

        # Compute fraud probability for nameOrig (using Laplace smoothing)
        orig_fraud_prob = (fraud_counts.get(orig, 0) + 1) / (total_counts.get(orig, 0) + 2)
        name_orig_probs.append(orig_fraud_prob)

        # Compute fraud probability for nameDest (using Laplace smoothing)
        dest_fraud_prob = (fraud_counts.get(dest, 0) + 1) / (total_counts.get(dest, 0) + 2)
        name_dest_probs.append(dest_fraud_prob)

        # Update total transaction count for each account
        total_counts[orig] = total_counts.get(orig, 0) + 1
        total_counts[dest] = total_counts.get(dest, 0) + 1

        # Update fraud transaction count for each account
        if is_fraud:
            fraud_counts[orig] = fraud_counts.get(orig, 0) + 1
            fraud_counts[dest] = fraud_counts.get(dest, 0) + 1

    # 10. Replace nameOrig and nameDest with their fraud probability
    df["nameOrig_fraud_prob"] = name_orig_probs
    df["nameDest_fraud_prob"] = name_dest_probs
    df.drop(columns=["nameOrig", "nameDest", "type"], inplace=True)
    df.drop(columns=["nameOrig_fraud_prob", "nameDest_fraud_prob"], inplace=True)

    return df


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
