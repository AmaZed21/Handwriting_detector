import tensorflow as tf
from tensorflow import keras #type: ignore
from tensorflow.keras import layers #type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.utils import class_weight
from scipy.io import loadmat
from zipfile import ZipFile

zip_path = 'ZIP PATH'
with ZipFile(zip_path, 'r') as zip_file:
    with zip_file.open('matlab/emnist-byclass.mat') as mat_file:
        mat = loadmat(mat_file)
#Loading data
x_train = mat['dataset'][0][0][0][0][0][0].reshape(-1, 28, 28)
y_train = mat['dataset'][0][0][0][0][0][1].flatten()
x_test = mat['dataset'][0][0][1][0][0][0].reshape(-1, 28, 28)
y_test = mat['dataset'][0][0][1][0][0][1].flatten()

weights = class_weight.compute_class_weight('balanced', 
                                            classes=np.unique(y_train), 
                                            y=y_train)
weight_dict = dict(enumerate(weights))
x_train = np.array([np.transpose(img) for img in x_train])
x_test = np.array([np.transpose(img) for img in x_test])

#Retrieving model metrics

history_df = pd.DataFrame(history.history)
history_df[['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()
history_df[['loss', 'val_loss']].plot()
plt.show()
