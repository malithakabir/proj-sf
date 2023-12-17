import os
import json
import pickle
from src.utils import Timer
from keras.layers import (
    Dense,
    Input,
    Activation,
    Dropout,
    RepeatVector,
    TimeDistributed,
    Flatten,
)
from keras.layers import LSTM
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD
import datetime as dt


class CustomModel:
    """
    Description:
    (1) use config_data to read data from pickle
    (2) use config_model to build model
    (3) use config_training  to set train model
    (4) use config_dir to save file
    """

    def __init__(self) -> None:
        self.model = Sequential()
        self.dataset = None
        self.training_history = None

    def build_model(self, input_n, output_n, drop_rate, latent_n, feature_n):
        """
        input_n: the length of the input sequence
        output_n: the length of the predicted sequence
        feature_n: how many features we have in the model
        latent_n: the size of the hidden units.
        """

        print("input_n", input_n)
        print("output_n", output_n)
        print("latent_n", latent_n)
        print("feature_n", feature_n)

        # =============================================================================
        # Bidirectional LSTM
        # =============================================================================

        #    3/26/2020

        encoder_inputs = Input(shape=(input_n, feature_n))

        # unidirectional LSTM layer
        encoder = LSTM(latent_n, return_state=True, activation="relu")

        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = RepeatVector(output_n)(encoder_outputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(
            latent_n, return_sequences=True, return_state=True, activation="relu"
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, initial_state=encoder_states
        )

        #    decoder_outputs_1 = Dense(output_n)(decoder_outputs)

        inter_output = Dropout(drop_rate)(decoder_outputs)

        decoder_outputs_2 = Dense(1)(inter_output)

        #    final_output = Dropout(drop_rate)(decoder_outputs)

        model = Model(encoder_inputs, decoder_outputs_2)

        # # Set the learning rate
        # learning_rate = 1e-5

        # # Set the optimizer
        # optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

        # Set the optimizer
        optimizer = SGD(momentum=0.9)

        #    start = time.time()
        model.compile(loss="mean_squared_error", optimizer=optimizer)

        #    print("Compilation Time:" + str(time.time()-start) )
        print(model.summary())

        self.model = model

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, modelpath):
        timer = Timer()
        timer.start()
        print("[Model] Training Started")
        print("[Model] %s epochs, %s batch size" % (epochs, batch_size))

        # Set the learning rate scheduler
        lr_schedule = LearningRateScheduler(
            lambda epoch: 1e-5 * 10**(epoch / 20) if epoch < 55 else 0.001
            )

        
        callbacks = [
            lr_schedule,
            EarlyStopping(monitor="val_loss", patience=10),
            ModelCheckpoint(
                filepath=modelpath, monitor="val_loss", save_best_only=True
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(X_test, y_test),
        )
        self.model.save(modelpath)

        # save history here
        self.log_training_history(
            history=history.history, fpath=modelpath.replace(".h5", ".json")
        )

        print("[Model] Training Completed. Model saved as %s" % modelpath)
        timer.stop()

    def log_training_history(self, history, fpath):
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def load_dataset(self, filepath=None):
        if filepath is None:
            raise ValueError("filepath must be provided")
        with open(filepath, "rb") as f:
            self.dataset = pickle.load(f)
        print("dataset loaded")
