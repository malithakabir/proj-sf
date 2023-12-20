from src.utils import FILENAME_PRESETS
import json
import pickle
import pandas as pd
from keras.models import load_model
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf

yf.pdr_override()


class DataLoader:
    """Loads data

    Intended to read data from  /data in project root
    Intended to export data to /data in project root

    Attributes
    ----------
    ticker: str, required
    df: pandas.DataFrame
    dateformat: str

    Methods
    -------
    __init__(ticker): Constructs DataLoader class
    read_local(filepath=None, isRawData=None): Reads raw or technical analysis data from /data folder in project root
    read_remote(until, since=None): Reads raw data from yahoo finance for preset ticker
    save_raw_data: Exports raw data read from yahoo finance
    """

    def __init__(self, ticker=None):
        """Construct DataLoader class

        Parameters
        ----------
        ticker: str, required, immutable, ticker symbol in upper case alphabets
        df: pandas.DataFrame, instantiated as None
        dateformat: str, immutable
        """
        if ticker is None:
            raise ValueError("cannot instantiate without ticker")
        self.ticker = ticker.upper()
        self.df = None
        self.dateformat = "%Y-%m-%d"

    def read_local(self, filepath=None, isRawData=None):
        """Reads data from /data folder, sets pandas.DataFrame to df placeholder

        Parameters
        ----------
        filepath: str, required, relative path to file, use as ./data/filename
        isRawdata: bool, required, set True for raw data and False otherwise
        """
        if filepath is None or isRawData is None:
            raise ValueError("filepath and isRawData flag must be provided")
        if isRawData:
            self.df = pd.read_csv(
                filepath, header=0, index_col="Date", parse_dates=True
            )
        else:
            self.df = pd.read_csv(filepath, parse_dates=True, index_col=0)

    def read_remote(self, until, since=None):
        """Reads raw data from yahoo finance for preset ticker

        Parameters
        ----------
        until: str, required, last date to be read
        since: str, required, first date to be read, 2006-12-13 is used is None parsed
        """
        if since is None:
            since = "2006-12-13"
        startdate = datetime.strptime(since, self.dateformat)
        enddate = datetime.strptime(until, self.dateformat)
        self.df = pdr.get_data_yahoo([self.ticker], start=startdate, end=enddate)

    def save_raw_data(self, version=None):
        """Exports raw data to /data directory of project root

        Paramters
        ---------
        version: int, required
        """
        if version is None:
            raise ValueError("must set data version")
        fpath = FILENAME_PRESETS["RAW"].format(self.ticker, version)
        self.df.to_csv(fpath, sep=",")
        print("raw data saved to : {}".format(fpath))

    def save_technical_analysis_data(self, df, ticker, version):
        """Exports technical analysis data to /data directory

        Parameters
        ----------
        df: pandas.DataFrame, required
        ticker: str, required
        version: int, required
        """
        # not logging here
        fpath = FILENAME_PRESETS["TA"].format(ticker, version)
        df.to_csv(fpath, sep=",")
        print("technical analysis data saved to : {}".format(fpath))

    def save_all_data_for_model_training(self, datadict, ticker, version):
        """Exports python objects

        Parameters
        ----------
        datadict: python dict, required
        ticker: str, required
        version: int, required
        """
        if ticker is None or version is None:
            raise ValueError("both ticker and version are requied")
        fpath = FILENAME_PRESETS["PREPROCESSED"].format(ticker, version)
        if isinstance(datadict, dict):
            with open(fpath, "wb") as f:
                pickle.dump(datadict, f)
            print("data saved")
        else:
            raise ValueError("data must be dict type object")


class ModelLoader:
    """Reads pretrained model and training history

    Methods
    -------
    read_model_local(basename = None): reads model from /model directory of project root
    read_training_history(basename = None): reads training history from /model directory of project root

    Attributes
    ----------
    basename: str, instantiated as None
    model: keras.Model, instantiated as None
    training_history: dict, instantiated as None
    """

    def __init__(self):
        self.basename = None
        self.model = None
        self.training_history = None

    def read_model_local(self, basename=None):
        """reads model specified by basename from /model directory of project root"""
        if basename is None:
            raise ValueError("cannot instantiate without basename")
        self.basename = basename
        print("Loading model from model/{}".format(self.basename + ".h5"))
        self.model = load_model(FILENAME_PRESETS["MODEL"].format(self.basename))

    def read_training_history(self, basename=None):
        """reads training history from /model directory specified by basename"""
        if basename is None:
            raise ValueError("cannot instantiate without basename")
        self.basename = basename
        fpath = FILENAME_PRESETS["TRAINING_HISTORY"].format(self.basename)
        with open(fpath, "r", encoding="utf-8") as f:
            self.training_history = json.load(f)
