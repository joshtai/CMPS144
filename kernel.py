# Los Alamos National Laboratory Earthquake Prediction

# The algorithm randomly takes clusters of data from the training set in 150,000-sample groups. 
# Each group of 150,000 values is then split into 150 pieces of length 1000. 
# Then we sample each piece in by looking at the entire 1000, only the last 100, 
# and only the last 10 values. We then extract 4 features (min, max, mean, std). 
# This results in a feature matrix of dimensions 150 time steps x 12 features. 

# import packages
import pandas as pd
import numpy as np 
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Set seeds
from numpy.random import seed
seed(4543)
from tensorflow import set_random_seed
set_random_seed(7509)

# Import the training data
train_data = pd.read_csv("../input/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})

# Visualize training data by sampling 2% of the data
train_acoustic_data_small = train_data['acoustic_data'].values[::50]
train_time_to_failure_small = train_data['time_to_failure'].values[::50]

fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")
plt.plot(train_acoustic_data_small, color='b')
ax1.set_ylabel('acoustic_data', color='b')
plt.legend(['acoustic_data'])
ax2 = ax1.twinx()
plt.plot(train_time_to_failure_small, color='g')
ax2.set_ylabel('time_to_failure', color='g')
plt.legend(['time_to_failure'], loc=(0.875, 0.9))
plt.grid(False)
plt.show()

del train_acoustic_data_small
del train_time_to_failure_small
train_data = train_data.values

# Helper function that extracts the max, min, standard deviation, and average per time step from a 2D array
def extract_features(z):
     return np.c_[z.max(axis=1), z.min(axis=1), z.std(axis=1), z.mean(axis=1)]

# For a given ending position "end_pos", we split the last 150'000 values 
# of "x" into 150 pieces of length 1000 each. So n_steps * step_len should equal 150'000.
# A set features is then extracted from each piece. 
# Returns a feature matrix with 150 time steps x features. 
def create_features(x, end_pos=None, n_steps=150, step_len=1000):
    if end_pos == None:
        end_pos = len(x)
       
    assert end_pos - n_steps * step_len >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    temp = (x[(end_pos - n_steps * step_len):end_pos].reshape(n_steps, -1) - 5 ) / 3
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    return np.c_[extract_features(temp),
                 extract_features(temp[:, -step_len // 10:]),
                 extract_features(temp[:, -step_len // 100:])]

# Query "create_features" to figure out the number of features
n_features = create_features(train_data[0:150000]).shape[1]
print("Our RNN is based on %i features"% n_features)
    
# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_features".
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_len=1000):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_len, max_index, size=batch_size)
         
        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        
        for i, row in enumerate(rows):
            samples[i] = create_features(data[:, 0], end_pos=row, n_steps=n_steps, step_len=step_len)
            targets[i] = data[row - 1, 1]
        yield samples, targets
        
batch_size = 32

# Position of second (of 16) earthquake. Used to have a clean split
# between train and validation
second_earthquake = 50085877
train_data[second_earthquake, 1]

# Initialize generators
train_generator = generator(train_data, batch_size=batch_size) # Use this for better score
validation_generator = generator(train_data, batch_size=batch_size, max_index=second_earthquake)

# Define model
from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential

cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]

model = Sequential()
model.add(CuDNNGRU(48, input_shape=(None, n_features)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile and fit model
model.compile(optimizer=adam(lr=0.0005), loss="mae")

history = model.fit_generator(train_generator,
                              steps_per_epoch=1000,
                              epochs=25,
                              verbose=0,
                              callbacks=cb,
                              validation_data=validation_generator,
                              validation_steps=200)

# Visualize loss
def perf_plot(history, what = 'loss'):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1
    
    plt.plot(epochs, x, 'bo', label = "Training " + what)
    plt.plot(epochs, val_x, 'b', label = "Validation " + what)
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    return None

perf_plot(history)

# Load submission file
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_features(x), 0))

submission.head()

# Save
submission.to_csv('submission.csv')