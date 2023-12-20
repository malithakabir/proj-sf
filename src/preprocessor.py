import talib as tb


class Preprocessor:
    """
    Description:
    It perform preprocessing
    Technical analysis can be performed if isRawData evaluates to True
    dropna command is required for before splitting training and test
    """

    def __init__(self):
        print("Instantiating preprocessor")

        self.df = None
        self.isRawData = None

    def set_df(self, df, isRawData):
        """
        Description: It set (1) dataframe (2) type of dataframe
        Notice: No technical analysis will be performed if isRawData evaluates to False or None, means - technical analysis already performed
        """
        self.df = df
        self.isRawData = isRawData
        print("setting df while isRawData={}".format(isRawData))

    def prepare_technical(self):
        """performs technical analysis

        Notice: incomplete docstring
        ----------------------------
        =============================================================================
         Prepare the features we use for the model
        =============================================================================
        1. HL_PCT: the variation of the stock price in a single day
        2. PCT_change: the variation between the open price and the close price
        3. Adj close price of the day
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
