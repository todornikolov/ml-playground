from datetime import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print("Tensorflow version: " + tf.__version__ + "\n")

# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb#scrollTo=gFh9ne3FZ-On
# First download and import the dataset using pandas:

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# The dataset contains a few unknown values:
dataset.isna().sum()

# Drop those rows to keep this initial tutorial simple:
dataset = dataset.dropna()

# Now, split the dataset into a training set and a test set.
# You will use the test set in the final evaluation of your models.

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
# Return a random sample of items from an axis of object.
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print("Initial dataset:")
print(train_dataset)

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

print()
print(train_dataset.describe().transpose())

# Separate the target value—the "label"—from the features.
# This label is the value that you will train the model to predict.
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

print("\n" + "Training features (removed MPG) dataset:")
print(train_features)

# Normalization:
# In the table of statistics it's easy to see how different the ranges of each feature are:
print()
print(train_dataset.describe().transpose()[['mean', 'std']])
print()

# The Normalization layer
# The tf.keras.layers.Normalization is a clean and simple way to add feature normalization into your model.
# The first step is to create the layer:
normalizer = tf.keras.layers.Normalization(axis=-1)

# Then, fit the state of the preprocessing layer to the data by calling Normalization.adapt:
normalizer.adapt(np.array(train_features))

# Calculate the mean and variance, and store them in the layer:
print(normalizer.mean.numpy())

# When the layer is called, it returns the input data, with each feature independently normalized:
first = np.array(train_features[:1])

print()
with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

# ---- Linear regression with one variable

# Begin with a single-variable linear regression to predict 'MPG' from 'Horsepower'.

# First, create a NumPy array made of the 'Horsepower' features.
# Then, instantiate the tf.keras.layers.Normalization and fit its state to the horsepower data:

horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build the Keras Sequential model:
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

print()
horsepower_model.summary()

# ----- This model will predict 'MPG' from 'Horsepower'.
# Run the untrained model on the first 10 'Horsepower' values.
# The output won't be good, but notice that it has the expected shape of (10, 1):
print(horsepower_model.predict(horsepower[:10]))

horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

plot_loss(history)

test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()

plot_horsepower(x,y)
