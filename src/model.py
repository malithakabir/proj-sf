import json
import pickle
from src.utils import Timer
from keras.layers import (
    Dense,
    Input,
    Dropout,
    RepeatVector,
)
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD


class CustomModel:
    """Constructs customized class for data reading, model training

    Description
    -----------
    (1) reads data for training
    (2) builds model
    (3) train model
    (4) export model
    (5) export training history
    """

    def __init__(self) -> None:
        """constructs CustomModel class

        Parameters
        ----------
        model: Keras Sequential model
        dataset: All required data
        history: model training history
        """
        self.model = Sequential()
        self.dataset = None
        self.history = None

    def build_model(self, input_n, output_n, drop_rate, latent_n, feature_n):
        """compile model

        Parameters
        ----------
        input_n: the length of the input sequence
        output_n: the length of the predicted sequence
        feature_n: how many features we have in the model
        latent_n: the size of the hidden units.
        """

        print("input_n", input_n)
        print("output_n", output_n)
        print("latent_n", latent_n)
        print("feature_n", feature_n)
        print("drop_rate", drop_rate)

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
        """train and export model, exports training history

        Notice
        ------
        TODO: docstring imporvement required

        Parameters
        ----------
        X_train:
        y_train:
        X_test:
        y_test:
        epochs: int, required
        batch_size: int, required
        modelpath: str, required, path to model in model directory
        """
        timer = Timer()
        timer.start()
        print("[Model] Training Started")
        print("[Model] %s epochs, %s batch size" % (epochs, batch_size))

        # Set the learning rate scheduler
        lr_schedule = LearningRateScheduler(
            lambda epoch: 1e-5 * 10 ** (epoch / 20) if epoch < 55 else 0.001
        )

        callbacks = [
            lr_schedule,
            EarlyStopping(monitor="val_loss", patience=10),
            ModelCheckpoint(
                filepath=modelpath, monitor="val_loss", save_best_only=True
            ),
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(X_test, y_test),
        )
        self.model.save(modelpath)

        # save history here
        fpath = modelpath.replace(".h5", ".json")
        with open(fpath, "w", encoding="utf-8") as f:
            self.history.history["lr"] = [float(x) for x in self.history.history["lr"]]
            json.dump(self.history.history, f, ensure_ascii=False, indent=4)

        print("[Model] Training Completed. Model saved as %s" % modelpath)
        timer.stop()

    def read_all_data_for_model_training(self, filepath=None):
        """read pickle file from /data directory of project root where pickle file contains all the necessary data"""
        if filepath is None:
            raise ValueError("filepath must be provided")
        with open(filepath, "rb") as f:
            self.dataset = pickle.load(f)
        print("dataset loaded")
