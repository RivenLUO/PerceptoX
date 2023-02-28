import os
import keras_tuner as kt
from tensorflow import keras
import numpy as np
import hp_models

# data preparation
folder_dir = r"D:\Zhewen\PerceptoX\data\datasets\train_data"
x_left_training = np.load(os.path.join(folder_dir, "train_left_duel_1.npy"), allow_pickle=True)
x_right_training = np.load(os.path.join(folder_dir, "train_right_duel_1.npy"), allow_pickle=True)
y_training = np.load(os.path.join(folder_dir, "train_label_duel_1.npy"), allow_pickle=True)

# 5780 samples: 5000 for training (hold 20% validation), 780 for testing
x_left_train, x_left_val = x_left_training[0:400], x_left_training[400:500]
x_right_train, x_right_val = x_right_training[0:400], x_right_training[400:500]

x_train = [x_left_train, x_right_train]
x_val = [x_left_val, x_right_val]
y_train, y_val = y_training[0:400], y_training[400:500]

hypermodel = hp_models.ComparisonHyperModel(num_classes=2)

save_dir = r"D:\Zhewen\PerceptoX\results"

tuner = kt.BayesianOptimization(
    hypermodel,
    objective=["val_accuracy"],
    max_trials=100,
    executions_per_trial=2,
    directory=os.path.join(save_dir, "hp_results_first_500_samples_2"),
    overwrite=True,
)

callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]

search_history = tuner.search(
    x_train,
    y_train,
    batch_size=8,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=2,
)

top_n = 4
best_hps = tuner.get_best_hyperparameters(top_n)
