import os
import numpy as np
import pandas as pd
import re
import json
from keras.models import load_model

# import talib as tb
from sklearn.preprocessing import StandardScaler
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf

yf.pdr_override()


class DataLoader:
    """
    Description: It uses log file to load data
    """

    def __init__(self, ticker=None):
        if ticker is None:
            raise ValueError("cannot instantiate without ticker")
        self.ticker = ticker
        self.df = None
        self.dateformat = "%Y-%m-%d"

    def read_local(self, filepath=None, isRawData=None):
        if filepath is None or isRawData is None:
            raise ValueError("filepath and isRawData flag must be provided")
        if isRawData:
            self.df = pd.read_csv(
                filepath, header=0, index_col="Date", parse_dates=True
            )
        else:
            self.df = pd.read_csv(filepath, parse_dates=True, index_col=0)

    def read_remote(self, until, since="2006-12-13"):
        startdate = since
        enddate = datetime.strptime(until, self.dateformat)
        self.df = pdr.get_data_yahoo([self.ticker], start=startdate, end=enddate)

    def save_raw_data(self, version=None):
        if version is None:
            raise ValueError("must set data version")
        fpath = "./data/{}_RAW_V{}.csv".format(self.ticker, version)
        self.df.to_csv(fpath, sep=",")


class ModelLoader:
    def __init__(self):
        self.basename = None
        self.model = None
        self.training_history = None

    def read_model_local(self, basename=None):
        if basename is None:
            raise ValueError("cannot instantiate without basename")
        self.basename = basename
        print("Loading model from {}".format(self.basename + ".h5"))
        self.model = load_model("model/{}.h5".format(self.basename))

    def read_training_history(self, basename=None):
        if basename is None:
            raise ValueError("cannot instantiate without basename")
        self.basename = basename
        fpath = "model/{}.json".format(self.basename)
        with open(fpath, 'r') as f:
            self.training_history = json.load(f)
