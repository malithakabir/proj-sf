import math
import numpy as np
from sklearn.metrics import mean_squared_error as mse


def invert_scale_N_feature(scaler, data, prediction):
    """
    Important: data includes the target and features
    Note: the first column in always the target i.e.: Adj Close
    """
    #   the 1st column is the 'Adj Close' price column
    any_seq_length = data.shape[0]
    pred_seq_length = prediction.shape[0]

    total_length_for_inversion = any_seq_length + pred_seq_length

    no_of_features = data.shape[1]

    result = np.zeros([total_length_for_inversion, no_of_features])

    result[:any_seq_length, :] = data  # insert data at top
    result[-pred_seq_length:, 0] = prediction  # insert predictions at bottom

    inverted = scaler.inverse_transform(result)

    inv_true = inverted[:-pred_seq_length, 0]
    inv_pred = inverted[-pred_seq_length:, 0]

    return inv_true, inv_pred


def calculate_rmse(scalers, data, labels, predicts, splitname):
    """
    Notice: splitname is manually provided, should come from the data itself, i.e.: test, training, validation
    """
    # invert the predicted and original test value to USD

    invert_predict = []  # predicted future test values (hold_days)
    invert_true = []  # true future test values (hold_days)
    actual = []  # actual test data in the previous days (seq_len)

    for i in range(len(predicts)):
        #    for i in range(1):
        _actual, inverted_pred = invert_scale_N_feature(
            scalers[i], data[i, :, :], predicts[i]
        )
        invert_predict.append(inverted_pred)

        _actual, inverted_true = invert_scale_N_feature(
            scalers[i], data[i, :, :], labels[i]
        )
        invert_true.append(inverted_true)
        actual.append(_actual)

    total_rmse = 0
    for i in range(len(invert_predict)):
        total_rmse = total_rmse + math.sqrt(mse(invert_predict[i], invert_true[i]))
    accuracy = total_rmse / len(invert_predict)
    print("The avearge root mse on %s data is: %.2f" % (splitname, accuracy))
    return accuracy
