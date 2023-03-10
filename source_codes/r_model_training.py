"""

"""
import os

import keras_tuner
import keras_tuner as kt
from tensorflow import keras
import numpy as np
import hp_models
import csv
import json
from kerastuner import HyperParameters
from train_test import simple_training

# data preparation------------------------------------------------------------------------------------------------------
# train_data_dir = r"D:\Zhewen\PerceptoX\data\datasets\train_data"
train_data_dir = r"E:\thesis MSc Geography\PerceptoX\data\datasets\train_data"
x_left_training = np.load(os.path.join(train_data_dir, "train_left_duel_1.npy"), allow_pickle=True)
x_right_training = np.load(os.path.join(train_data_dir, "train_right_duel_1.npy"), allow_pickle=True)
y_training = np.load(os.path.join(train_data_dir, "train_label_duel_1.npy"), allow_pickle=True)

# 5780 samples: the first 5000 for training (hold 20% validation), the rest 780 for testing!!!
x_left_training = x_left_training[0:5000]
x_right_training = x_right_training[0:5000]
x_train = [x_left_training, x_right_training]
y_train = y_training[0:5000]

# Mean centered due to VGG19 was trained on mean-centered ImageNet (B:103.939,G:116.779,R:223.68)
bgr_mean_Imagenet = np.array([103.939, 116.779, 123.68])
x_left_training -= bgr_mean_Imagenet.reshape((1, 1, 1, 3))
x_right_training -= bgr_mean_Imagenet.reshape((1, 1, 1, 3))

# model configuration---------------------------------------------------------------------------------------------------
save_dir = r"D:\Zhewen\PerceptoX\results"
save_best_hp_dir = "hp_results_Q1_1"
with open(os.path.join(save_dir, save_best_hp_dir, f"best_1_hps_config.json"), 'r') as f:
    hps_config = json.load(f)
best_hps = HyperParameters.from_config(hps_config)
tuner = keras_tuner.Tuner()
model = tuner.hypermodel.build(best_hps)

# Train model-----------------------------------------------------------------------------------------------------------
simple_training(train_data=x_train, train_label=y_train, val_split=0.2,
                model_config=model, epochs=50, batch_size=8)
