import datetime as dt

FILENAME_PRESETS = {
    "RAW": "./data/{}_RAW_V{}.csv",
    "TA": "data/{}_TA_V{}.csv",
    "MODEL": "model/{}.h5",
    "TRAINING_HISTORY": "model/{}.json",
    "PREPROCESSED": "data/{}_PREPROCESSED_V{}.pickle",
}


class Timer:
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print("Time taken: %s" % (end_dt - self.start_dt))


def print_data_split(df_train, df_test, df_predict):
    print("Training data starts at", df_train.index[0])
    print("Training - test split at", df_train.index[-1])
    print("Testing data ends at", df_test.index[-1])
    print()
    print("prediction data starts at", df_predict.index[0])
    print("prediction data ends at", df_predict.index[-1])
