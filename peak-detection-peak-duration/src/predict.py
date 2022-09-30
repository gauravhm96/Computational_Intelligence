import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import sys
sys.path.append('../src/cnn')

from models import ConvNet
from preprocessing import LabelEncoder
import argparse

import tkinter as tk
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def read_data(path, header=None, sep=',', row_num = 1):
    from scipy.interpolate import interp1d
    df = pd.read_csv(path, header=None, sep=',')
    
    # User list comprehension to create a list of lists from Dataframe rows
    data_list_of_rows = [list(row) for row in df.values]
    
    num = int(row_num)
    y_axis = data_list_of_rows[num]
    
    print("You have chosen signal:", num);
    
    column_in = len(y_axis)
    col = range(0,column_in)
    x_axis = list(col)
    
    elon_list = [11, 21, 19, 18, 29]
    data_x = np.asarray(x_axis)
    data_y = np.asarray(y_axis)

    #return data_x, data_y


    # Obtain interpolation function (input original x (time) and y (signal))
    f = interp1d(data_x, data_y)
    # Create new x (same x_min and x_max but different number of data points)
    x_new = np.linspace(data_x.min(), data_x.max(), 8192)
    # Obtain new y (based on new x)
    y_new = f(x_new)
    # return both new x and new y
    return x_new, y_new


def predict_peek_duration(input_data_file,row_number):
    NUM_TRAIN_EXAMPLES = int(1e6)
    NUM_TEST_EXAMPLES = int(1e4)

    NUM_CLASSES = 3
    NUM_WINDOWS = 256
    INPUT_SIZE = 8192

    # Define label encoder to decode predictions later on
    label_encoder = LabelEncoder(NUM_WINDOWS)

    # Build model (should have the same parameters as the model
    # in 01_cnn-train.ipynb)
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
    # If retrained (via 01_train-cnn.ipynb), weights should be loaded as 
    # outputs/weights/weight_{epoch:03d}.h5 (e.g. outputs/weights/weight_009.h5)
    model.load_weights('../src/output/weights/cnn_weights.h5')


    # Read data (applies linear interpolation to get the right dimension)
    # t = time; s = signal
    #t, s = read_data('../input/training.csv',header=None, sep=',')

    t, s = read_data(input_data_file,header=None, sep=',', row_num = row_number)



    # Normalize and add batch dimension (model expect batched data)
    s_norm = s[None, :, None] / s.max()


    # Pass to model and make predictions
    preds = model(s_norm)[0]

    # .decode will filter out predictions below threshold
    # and compute the appropriate locations of the peaks
    probs, locs, areas = label_encoder.decode(preds, threshold=0.5)

    print("Predicted locations:\n", locs * t.max())
    print("\nPredicted areas:\n", areas)
    print("\nPredicted probabilities:\n", probs)


    peakdetects    = np.asarray(locs * t.max())

    print("\ninitialpeakdetect:\n",peakdetects[0])
    print("\nlastpeakdetect:\n",peakdetects[-1])
    print("\nPulse Duration(msec) :\n",peakdetects[-1] - peakdetects[0])

    root= tk.Tk() 
    # Visualize locations in the chromatogram
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    bar1 = FigureCanvasTkAgg(fig, root)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    ax.plot(t, s)

    for (prob, loc, area) in zip(probs, locs, areas):
        y = s.min() - s.max() * 0.05 # location of the triangles
        ax.scatter(loc*t.max(), y, color='C1', marker='^', s=100, edgecolors='black')
        
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title('Peak Detection and Pulse Duration Estimation')
    ax.set_ylabel('Signal', fontsize=16)
    ax.set_xlabel('Time Scale(msec)', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    root.mainloop()

if __name__ == '__main__':
    

    print (sys.argv[1],sys.argv[2])
    if (len(sys.argv) > 1):
        parser = argparse.ArgumentParser(description="less script")
        parser.add_argument('--input_file', required=True, help="input file containing IDs and attributes to change (csv)")
        parser.add_argument('--row_num', required=True, help="row number to select signal")
        args = parser.parse_args()
        
        if args.input_file is not None:
            process_data_file = args.input_file
        if args.input_file is not None:
            row_number = args.row_num
    
    else:
        print("Invalid number of arguments. It accepts only one argument, which is input_file")
        sys.exit()
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    predict_peek_duration(process_data_file,row_number)
