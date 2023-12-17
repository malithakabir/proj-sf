import pickle
import numpy as np
import talib as tb
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Description:
    It perform preprocessing
    Technical analysis can be performed if isRawData evaluates to True
    dropna command is required for before splitting training and test
    """

    def __init__(self):
        print("Instantiating preprocessor")

        # required variable during runtime, must be invoked
        self.isnadrop = False
        self.df = None
        self.isRawData = None

        # these needs to be set
        self.rolling_window = -1
        self.forecast_horizon = -1
        self.test_size = -1
        
        # Not Implemented properly
        self.sequence_length = None
        self.sample_size_mt = None
        self.index_of_tt_split = None

    def set_df(self, df, isRawData):
        """
        Description: It set (1) dataframe (2) type of dataframe
        Notice: No technical analysis will be performed if isRawData evaluates to False or None, means - technical analysis already performed
        """
        self.df = df
        self.isRawData = isRawData

    def dropna(self):
        """
        Description: It drops nan values
        Notice: Parent dataframe is replace by this operation
        """
        self.isnadrop = True
        self.df = self.df.dropna(axis=0).copy()

    def set_rw_fh_test_size(self, rw, fh, test_size):
        """
        Description: Sets rolling window and forecast horizons
        """
        self.rolling_window = rw
        self.forecast_horizon = fh
        self.test_size = test_size
        
    def generate_train_test_predict_split(self):
        """
        Required:
            df
            rolling_window
            test_size
        Returns: dict [df_train, df_test, df_predict]
        """
        if self.isnadrop:
            df_predict = self.df.iloc[
                -self.rolling_window :
            ]  # for future forecasting use
            df_train_test = self.df.iloc[
                : -self.rolling_window
            ]  # for model training and testing use

            # Need to export in file
            self.sample_size_mt = df_train_test.shape[0]
            self.index_of_tt_split = round((1 - self.test_size) * self.sample_size_mt)

            df_train = df_train_test.iloc[: self.index_of_tt_split, :]
            df_test = df_train_test.iloc[self.index_of_tt_split :, :]

            # # print results
            # print_data_split(df_train, df_test, df_predict)

            return {"df_train": df_train, "df_test": df_test, "df_predict": df_predict}
        else:
            print("Drop NA first using dropna() method")

    def normalise_dataframe(self, df, step=1, standard_norm=True):
        """
        Description:
        (1) It creates scalers on the sequence lengh
        (2) It apply scalers on the respective sequences
        (3) It returns normalized data that can be used in preparing features and labels
        """
        # # standard_norm is always true

        self.sequence_length = self.rolling_window + self.forecast_horizon

        if standard_norm:
            normalised_data = []
            scalers = []
            # form a sliding window of size 'sequence_length', until the data is exhausted
            for index in range(0, df.shape[0] - self.sequence_length + 1, step):
                window = df[index : index + self.sequence_length]
                scaler = StandardScaler()
                scaler.fit(window)
                normalised_data.append(scaler.transform(window))
                scalers.append(scaler)
            return {"normalised_data": normalised_data, "scalers": scalers}

    def prepare_feature_and_label(self, data_list):
        """
        Description: It prepares features and labels from normalised data
        Notice: Normalised data is list type
        """
        arr = np.array(data_list)

        # shape: sample, timeframe, features
        features = arr[
            :, : -self.forecast_horizon, 1:
        ]  # features: everything except the first col

        labels = arr[:, -self.forecast_horizon :, 0]  # labels: the first col only
        # reshape label to sample, timeframe, label
        labels = np.reshape(labels, (labels.shape[0], labels.shape[1], 1))

        return {"data": arr, "features": features, "labels": labels}

    def prepare_technical(self):
        """
        # =============================================================================
        #  Prepare the features we use for the model
        # =============================================================================
        #1. HL_PCT: the variation of the stock price in a single day
        #2. PCT_change: the variation between the open price and the close price
        #3. Adj close price of the day
        """
        if self.isRawData:
            df_technical = self.df.copy()
            # transform the data and get %change daily
            df_technical["HL_PCT"] = (
                (df_technical["High"] - df_technical["Low"])
                / df_technical["Low"]
                * 100.0
            )
            # spread/volatility from day to day
            df_technical["PCT_change"] = (
                (df_technical["Adj Close"] - df_technical["Open"])
                / df_technical["Open"]
                * 100.0
            )

            # obtain the data from the technical analysis function and process it into useful features
            # open = df_technical['Open'].values
            close = df_technical["Adj Close"].values
            high = df_technical["High"].values
            low = df_technical["Low"].values
            volume = df_technical["Volume"].values

            # The technical indicators below cover different types of features:
            # 1) Price change – ROCR, MOM
            # 2) Stock trend discovery – ADX, MFI
            # 3) Buy&Sell signals – WILLR, RSI, CCI, MACD
            # 4) Volatility signal – ATR
            # 5) Volume weights – OBV
            # 6) Noise elimination and data smoothing – TRIX

            # define the technical analysis matrix

            #        https://www.fmlabs.com/reference/default.htm?url=ExpMA.htm

            #  Overlap Studies

            # make sure there is NO forward looking bias.
            # moving average
            df_technical["MA_5"] = tb.MA(close, timeperiod=5)
            df_technical["MA_20"] = tb.MA(close, timeperiod=20)
            df_technical["MA_60"] = tb.MA(close, timeperiod=60)
            # df_technical['MA_120'] = tb.MA(close, timeperiod=120)

            # exponential moving average
            df_technical["EMA_5"] = tb.MA(close, timeperiod=15)
            # 5-day halflife. the timeperiod in the function is the "span".

            df_technical["up_band"] = tb.BBANDS(
                close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
            )[0]

            df_technical["mid_band"] = tb.BBANDS(
                close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
            )[1]

            df_technical["low_band"] = tb.BBANDS(
                close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
            )[2]

            # Momentum Indicators
            df_technical["ADX"] = tb.ADX(high, low, close, timeperiod=20)
            # df_technical['ADXR'] = tb.ADXR(high, low, close, timeperiod=20)

            df_technical["MACD"] = tb.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )[2]
            df_technical["RSI"] = tb.RSI(close, timeperiod=14)
            # df_technical['AD'] = tb.AD(high, low, close, volume)
            df_technical["ATR"] = tb.ATR(high, low, close, timeperiod=14)
            df_technical["MOM"] = tb.MOM(close, timeperiod=10)
            df_technical["WILLR"] = tb.WILLR(high, low, close, timeperiod=10)
            df_technical["CCI"] = tb.CCI(high, low, close, timeperiod=14)

            #   Volume Indicators
            df_technical["OBV"] = tb.OBV(close, volume * 1.0)

            # # drop the NAN rows in the dataframe
            # df3.dropna(axis = 0, inplace = True)
            # df3.fillna(value=-99999, inplace=True)

            df3 = df_technical[
                [
                    "Adj Close",
                    "HL_PCT",
                    "PCT_change",
                    "Volume",
                    "MA_5",
                    "MA_20",
                    "MA_60",
                    #                   'MA_120',
                    "EMA_5",
                    "up_band",
                    "mid_band",
                    "low_band",
                    "ADX",
                    "MACD",
                    "RSI",
                    "ATR",
                    "MOM",
                    "WILLR",
                    "CCI",
                    "OBV",
                ]
            ]

            # forecast_col = 'Adj Close'
            # df3.loc[:,'label'] = df_technical[forecast_col].shift(-forecast_out)

            return df3

    def save_technical_analysis(self, df, ticker, version):
        # not logging here
        fpath = "data/{}_TA_V{}.csv".format(ticker, version)
        df.to_csv(fpath, sep=",")
        print("technical analysis data saved to : {}".format(fpath))

    def save_all_data_for_model_training(self, datadict, ticker, version):
        if ticker is None or version is None:
            raise ValueError("both ticker and version are requied")
        fpath = "data/{}_PREPROCESSED_V{}.pickle".format(ticker, version)
        with open(fpath, "wb") as f:
            pickle.dump(datadict, f)
        print("data saved")
