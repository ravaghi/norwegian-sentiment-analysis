import sys

sys.path.append('../')

import data.norec.dataloader as dataloader
from utils.visualization import plot_history
from data.dataloader import load_data
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping
from keras_tuner import Hyperband, RandomSearch
import os
import pickle
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(hp):
    """Builds a model with the hyperparameters.
    Args:
        hp: hyperparameter object

    Returns: Compiled model
    """

    # Defining hyperparameters
    hp_embedding_dim = hp.Int("embedding_dim", min_value=8, max_value=128, step=8)
    hp_lstm_units = hp.Int("lstm_units", min_value=8, max_value=128, step=128)
    hp_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05)
    hp_spatial_dropout = hp.Float("spatial_dropout", min_value=0.0, max_value=0.5, step=0.05)
    hp_l1_reg = hp.Choice("l1_regularizer", values=[0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.012, 0.015, 0.017, 0.02])
    hp_l2_reg = hp.Choice("l2_regularizer", values=[0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.012, 0.015, 0.017, 0.02])
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4])

    # Building model
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=hp_embedding_dim, input_length=maxlen))
    model.add(SpatialDropout1D(hp_spatial_dropout))
    model.add(LSTM(units=hp_lstm_units, dropout=hp_dropout, kernel_regularizer=l1_l2(l1=hp_l1_reg, l2=hp_l2_reg)))
    model.add(Dense(units=num_classes, activation="softmax"))
    model.compile(optimizer=adam_v2.Adam(hp_learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    RANDOM = False
    PROJECT_NAME = "random_search_hps_tuner" if RANDOM else "hyperband_hps_tuner"

    EPOCHS = 20
    BATCH_SIZE = 256

    # Loading data
    dataset = dataloader.load_full_dataset()
    processed_data = load_data(dataset)
    X_train = processed_data["X_train"]
    X_val = processed_data["X_val"]
    X_test = processed_data["X_test"]
    y_train = processed_data["y_train"]
    y_val = processed_data["y_val"]
    y_test = processed_data["y_test"]
    maxlen = processed_data["maxlen"]
    num_classes = processed_data["num_classes"]
    num_words = processed_data["num_words"]

    # Early stopping
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5),
        EarlyStopping(monitor="val_loss", patience=5),
    ]

    # Specifying tuner
    if RANDOM:
        tuner = RandomSearch(
            build_model,
            objective="val_accuracy",
            max_trials=1000,
            executions_per_trial=3,
            directory=BASE_DIR,
            project_name=PROJECT_NAME
        )
    else:
        tuner = Hyperband(
            build_model,
            objective="val_accuracy",
            max_epochs=1000,
            factor=3,
            directory=BASE_DIR,
            project_name=PROJECT_NAME,
        )

    # Training and searching
    tuner.search(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # Saving results to file
    with open(f"{PROJECT_NAME}/tuner.pkl", "wb") as f:
        pickle.dump(tuner, f)
    tuner = pickle.load(open(f"{PROJECT_NAME}/tuner.pkl", "rb"))

    # Loading the best model
    best_hps = tuner.get_best_hyperparameters(1)[0]

    # Saving hyperparameters of the best model
    with open(f"{PROJECT_NAME}/best_hps.json", "w") as f:
        json.dump(best_hps.get_config(), f)

    # Loading the best model and training it on the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val), callbacks=callbacks)

    # Saving the best model and plotting the results
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=1)
    model.save(f"models/best_model_{val_acc}.h5")
    plot_history(history, f"best_model_{val_acc}")
