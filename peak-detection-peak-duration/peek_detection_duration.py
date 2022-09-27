#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:41:24 2022

@author: kirankumarathirala
"""
import argparse
import sys
# Import all components from tkinter library for simple GUI
import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import multiprocessing
import sys
import logging
import glob

tf.get_logger().setLevel(logging.ERROR)

sys.path.append('src/cnn')


# import modules
from simulator import Simulator
from models import ConvNet
from losses import CustomLoss
from callbacks import SaveModelWeightsCallback

from metrics import CustomAUC
from metrics import CustomMREArea
from metrics import CustomMAELoc
from metrics import get_accuracy_metrics_at_thresholds

from preprocessing import LabelEncoder
from data_generators import DataGenerator

from models import ConvNet
from preprocessing import LabelEncoder

def model_training():
    # Model training
    NUM_TRAIN_EXAMPLES = int(1e6)
    NUM_TEST_EXAMPLES = int(1e4)
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 1e5 // BATCH_SIZE
    NUM_EPOCHS = 10

    # Model optimizer
    INITIAL_LEARNING_RATE = 1e-3
    END_LEARNING_RATE = 1e-5
    DECAY_STEPS = int(STEPS_PER_EPOCH / BATCH_SIZE * NUM_EPOCHS)

    # Label encoder
    NUM_CLASSES = 3
    NUM_WINDOWS = 256
    INPUT_SIZE = 8192

    # Define loss function, optimizer and metrics
    loss_fn = CustomLoss(
        n_splits=NUM_CLASSES, 
        weight_prob=1.0, 
        weight_loc=1.0, 
        weight_area=1.0
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=INITIAL_LEARNING_RATE, 
            decay_steps=DECAY_STEPS, 
            end_learning_rate=END_LEARNING_RATE
        )
    )

    metrics = [
        CustomMREArea(name='mre_area'),
        CustomMAELoc(name='mae_loc'),
        CustomAUC(name='prob_auc'),
    ]

    # Define data generators 
    # Simulator uses method of label_encoder to deal with collisions
    label_encoder = LabelEncoder(NUM_WINDOWS)
    train_generator = DataGenerator(
        indices=np.arange(0, NUM_TRAIN_EXAMPLES), 
        simulator=Simulator(
            label_encoder.remove_collision, 
            resolution=INPUT_SIZE, 
            white_noise_prob=1.0), 
        label_encoder=label_encoder,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_generator = DataGenerator(
        indices=np.arange(NUM_TRAIN_EXAMPLES, NUM_TRAIN_EXAMPLES + NUM_TEST_EXAMPLES), 
        simulator=Simulator(
            label_encoder.remove_collision, 
            resolution=INPUT_SIZE, 
            white_noise_prob=1.0), 
        label_encoder=label_encoder,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ConvNet(
        filters=[64, 128, 128, 256, 256],
        kernel_sizes=[9, 9, 9, 9, 9],
        dropout=0.0,
        pool_type='max',
        pool_sizes=[2, 2, 2, 2, 2],
        conv_block_size=1,
        input_shape=(INPUT_SIZE, 1),
        output_shape=(NUM_WINDOWS, NUM_CLASSES),
        residual=False
    )

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

    # Weights will be saved each epoch to outputs/weights/weight_{epoch:03d}.h5
    model.fit(
        train_generator, validation_data=test_generator,
        epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=[SaveModelWeightsCallback()]
    )
def read_data(path):
    from scipy.interpolate import interp1d
    
    # User list comprehension to create a list of lists from Dataframe rows
    csv_data = pd.read_excel(path)
    

    num = int(input("input a value between 1 to 531 to choose one signal amongst the 3 signals: "))
    y_axis = csv_data.iloc[num , 17:]
    
    print("You have chosen signal:", num);
    
    column_in = len(y_axis)
    col = range(0,column_in)
    x_axis = list(col)
    
    elon_list = [11, 21, 19, 18, 29]
    data_x = np.asarray(x_axis)
    data_y = np.asarray(y_axis)

    # Obtain interpolation function (input original x (time) and y (signal))
    f = interp1d(data_x, data_y)
    # Create new x (same x_min and x_max but different number of data points)
    x_new = np.linspace(data_x.min(), data_x.max(), 8192)
    # Obtain new y (based on new x)
    y_new = f(x_new)
    # return both new x and new y
    return x_new, y_new
def predict_peeks(data_file_path):
    NUM_TRAIN_EXAMPLES = int(1e6)
    NUM_TEST_EXAMPLES = int(1e4)

    NUM_CLASSES = 3
    NUM_WINDOWS = 256
    INPUT_SIZE = 8192

    # Define label encoder to decode predictions later on
    label_encoder = LabelEncoder(NUM_WINDOWS)

    # Build model (should have the same parameters as the model

    model = ConvNet(
        filters=[64, 128, 128, 256, 256],
        kernel_sizes=[9, 9, 9, 9, 9],
        dropout=0.0,
        pool_type='max',
        pool_sizes=[2, 2, 2, 2, 2],
        conv_block_size=1,
        input_shape=(INPUT_SIZE, 1),
        output_shape=(NUM_WINDOWS, NUM_CLASSES),
        residual=False
    )

    # Load pretrained weights
    # If retrained, weights should be loaded as 
    # outputs/weights/weight_{epoch:03d}.h5 (e.g. outputs/weights/weight_009.h5)
    model.load_weights('output/weights/cnn_weights.h5')
    # Read data (applies linear interpolation to get the right dimension)
    # t = time; s = signal
    t, s = read_data(data_file_path)

    # Normalize and add batch dimension (model expect batched data)
    s_norm = s[None, :, None] / s.max()


    # Pass to model and make predictions
    preds = model(s_norm)[0]

    # .decode will filter out predictions below threshold
    # and compute the appropriate locations of the peaks
    probs, locs, areas = label_encoder.decode(preds, threshold=0.5)

    print("Predicted locations:\n", locs * t.max())
    # locs * t.max()[-1] - locs * t.max()[0]
    print("\nPredicted areas:\n", areas)
    print("\nPredicted probabilities:\n", probs)
    
    # Visualize locations in the chromatogram
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(t, s)

    for (prob, loc, area) in zip(probs, locs, areas):
        y = s.min() - s.max() * 0.05 # location of the triangles
        ax.scatter(loc*t.max(), y, color='C1', marker='^', s=100, edgecolors='black')
        
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylabel('Signal', fontsize=16)
    ax.set_xlabel('Time', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xlim(0.0, 3.0)
    
if __name__ == '__main__':
    
    process_data_file = None
    
    if (len(sys.argv) == 1):
        parser = argparse.ArgumentParser(description="less script")
        parser.add_argument('--input_file', required=False, help="input file containing IDs and attributes to change (csv)")
        args = parser.parse_args()
        
        if args.input_file is not None:
            process_data_file = args.input_file
        
    else:
        print("Invalid number of arguments. It accepts only one argument, which is input_file")
        sys.exit()
    if process_data_file is None:
        
        try:
        
           root = tk.Tk()
           root.wm_withdraw()
           
        
           process_data_file = filedialog.askopenfilename(initialdir = "/",
                                         filetypes = (("CSV files",
                                                       "*.xlsx*"),
                                                      ("all files",
                                                       "*.*")),
                                         title = "Choose Training CSV File.")
           root.update()
           root.destroy()
           root.mainloop()
           
           model_training()
           
           physical_devices = tf.config.list_physical_devices('GPU')
           try:
               tf.config.experimental.set_memory_growth(physical_devices[0], True)
           except:
               pass
           
           
           predict_peeks(process_data_file)
        except Exception as ex:
            print("Exception",ex)
        