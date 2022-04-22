import data.norec.dataloader as dataloader
from data.dataloader import load_data
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(hp):
    hp_embedding_dim = hp.Int("embedding_dim", min_value=5, max_value=100, step=5)
    hp_lstm_units = hp.Int("lstm_units", min_value=8, max_value=128, step=128)
    hp_spatial_dropout = hp.Float("spatial_dropout", min_value=0.0, max_value=0.5, step=0.05)
    hp_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05)
    hp_l1_reg = hp.Choice("l1_regularizer", values=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1])
    hp_l2_reg = hp.Choice("l2_regularizer", values=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1])
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4])

    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=hp_embedding_dim, input_length=maxlen))
    model.add(SpatialDropout1D(rate=hp_spatial_dropout))
    model.add(LSTM(units=hp_lstm_units, dropout=hp_dropout, kernel_regularizer=l1_l2(l1=hp_l1_reg, l2=hp_l2_reg)))
    model.add(Dense(units=num_classes, activation="softmax"))
    model.compile(optimizer=adam_v2.Adam(hp_learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    EPOCHS = 20
    BATCH_SIZE = 128

    dataset = dataloader.load_full_dataset()
    processed_data = load_data(dataset)
    X_train = processed_data["X_train"]
    X_val = processed_data["X_val"]
    y_train = processed_data["y_train"]
    y_val = processed_data["y_val"]
    maxlen = processed_data["maxlen"]
    num_classes = processed_data["num_classes"]
    num_words = processed_data["num_words"]

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5),
        EarlyStopping(monitor="val_loss", patience=5),
    ]

    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=100,
        executions_per_trial=1,
        directory=BASE_DIR,
        project_name="sena_tuner",
    )

    print(tuner.search_space_summary())

    tuner.search(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    with open(f"tuner_results.pkl", "wb") as f:
        pickle.dump(tuner, f)

    """tuner = pickle.load(open("tuner_results.pkl", "rb"))
    print(tuner.get_best_hyperparameters()[0].values)
    print(tuner.results_summary())
    print(tuner.get_best_models()[0].summary())"""
