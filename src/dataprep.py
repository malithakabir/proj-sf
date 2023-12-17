import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPrep:
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
        Input: takes normalised data
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

    def save_all_data_for_model_training(self, datadict, ticker, version):
        if ticker is None or version is None:
            raise ValueError("both ticker and version are requied")
        fpath = "data/{}_PREPROCESSED_V{}.pickle".format(ticker, version)
        with open(fpath, "wb") as f:
            pickle.dump(datadict, f)
        print("data saved")
