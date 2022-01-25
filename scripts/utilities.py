import numpy as np
import pandas as pd
import os
import copy
import glob
import random
import pickle
import spacy
import re
import string
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, BertConfig

from sklearn.model_selection import train_test_split, KFold

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import seaborn as sns
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

from tqdm.autonotebook import tqdm 


def get_loader(dataframe, tokenizer, mode="train", max_length=512):
    """
    Retrieve DataLoader object for a given dataframe (e.g. train_df).
    """
    dataset = MedDataset(dataframe, tokenizer, mode, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True if mode == "train" else False, num_workers=0)
    return dataloader


def load_data_as_dfs(train_size=0.9, val_size=0.1, seed=111):
    """
    Load MT samples data as pandas dataframes
    """
    data = pd.read_csv("/content/drive/MyDrive/project/data/X.csv")
    data['label'] = (data['label'] - 1).copy()

    # Train-test split
    df_train=data.sample(frac=train_size,random_state=seed)
    df_test = data.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    # train-validation split
    val_frac = val_size / train_size
    df_val = df_train.sample(frac=val_frac,random_state=seed)
    df_train = df_train.drop(df_val.index).reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # print shapes
    print("Train:", df_train.shape)
    print("Val:", df_val.shape)
    print("Test:", df_test.shape)

    return df_train, df_val, df_test

def get_model_tokenizer(best_model_path, num_labels=4):
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # grab model name and dropout
    model_name = best_model_path.split("model_")[1].split("_")[0]
    dropout = float(best_model_path.split("dropout_")[1].split("_")[0])

    # Load empty model + tokenizer, fill with best model weights 
    if model_name == "bert-base-uncased":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = BERTClassifier(num_labels=num_labels, dropout=dropout).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    elif model_name == 'COReClassifier':
        tokenizer = AutoTokenizer.from_pretrained("bvanaken/CORe-clinical-diagnosis-prediction")
        model = COReClassifier(num_labels=num_labels, dropout=dropout).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    else:
      print("Invalid model_name!")

    return model, tokenizer


def test_model_acc(model, tokenizer, test_dataframe, column_name='text'):  

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataframe
    test_loader = get_loader(test_dataframe, tokenizer, mode="test")

    # evaluation mode
    model.eval()
    all_preds = None
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(batch)
            if all_preds is None:
                all_preds = preds
            else:
                all_preds = torch.cat([all_preds, preds], dim=0)
    
    # final model class predictions
    predictions = all_preds.argmax(dim=1).cpu().numpy()

    # final accuracy...
    test_acc = (predictions == test_dataframe['label'].values).mean()
    
    # AUC? Confusion Matrix? Other stuff?
    ####

    return predictions, test_acc