import os
import keras_tuner as kt
from tensorflow import keras
import numpy as np
import hp_models
import csv

# data preparation
folder_dir = r"D:\Zhewen\PerceptoX\data\datasets\train_data"
x_left_training = np.load(os.path.join(folder_dir, "train_left_duel_1.npy"), allow_pickle=True)
x_right_training = np.load(os.path.join(folder_dir, "train_right_duel_1.npy"), allow_pickle=True)
y_training = np.load(os.path.join(folder_dir, "train_label_duel_1.npy"), allow_pickle=True)

# 5780 samples: 5000 for training (hold 20% validation), 780 for testing
x_left_train, x_left_val = x_left_training[0:4000], x_left_training[4000:5000]
x_right_train, x_right_val = x_right_training[0:4000], x_right_training[4000:5000]

# Mean centered due to VGG19 was trained on mean-centered ImageNet (B:103.939,G:116.779,R:223.68)
bgr_mean_Imagenet = np.array([103.939, 116.779, 123.68])
x_left_train -= bgr_mean_Imagenet.reshape((1, 1, 1, 3))
x_right_train -= bgr_mean_Imagenet.reshape((1, 1, 1, 3))

x_train = [x_left_train, x_right_train]
x_val = [x_left_val, x_right_val]
y_train, y_val = y_training[0:4000], y_training[4000:5000]

# number of class is 3 : left, right, no reference y.shape=(5780,2)
hypermodel = hp_models.ComparisonHyperModel(num_classes=2)

save_dir = r"D:\Zhewen\PerceptoX\results"
save_best_hp_dir = "hp_results_Q1_1"

tuner = kt.BayesianOptimization(
    hypermodel,
    objective=["val_accuracy"],
    max_trials=100,
    executions_per_trial=2,
    directory=os.path.join(save_dir, save_best_hp_dir),
    overwrite=True,
)

callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]

try:
    search_history = tuner.search(
        x_train,
        y_train,
        batch_size=8,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )
except RuntimeError:
    # Skip the current trial and move to the next one
    print("RuntimeError: Number of consecutive failures exceeded the limit of 3.")

top_n = 5
best_hps = tuner.get_best_hyperparameters(top_n)

print(best_hps[0].values)

# save the best 5 hps as csv
for i in range(len(best_hps)):
    with open(os.path.join(save_dir, save_best_hp_dir, f"best_{i}_hps.csv"), "w") as file:
        # Create a writer object
        writer = csv.DictWriter(file, fieldnames=best_hps[i].values.keys())

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerow(best_hps[i].values)
        print('Done writing best_hps dict to a csv file')

