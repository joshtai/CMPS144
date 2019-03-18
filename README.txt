README

We first load and visualize the data by sampling every 50th value (2% of total data). Then we process the dataset by randomly selecting clusters of data from the training set in 150,000-sample groups. Each group of 150,000 values is then split into 150 pieces of length 1000. Then we sample each piece in by looking at the entire 1000, only the last 100, and only the last 10 values. We then extract 4 features (min, max, mean, std). This results in a feature matrix of dimensions 150 time steps x 12 features.

We define our model with a single GRU layer that had 48 nodes and 2 Dense layers, one with 10 node, and the last with a single output node.  When compiling the model we used adam for our optimizer and for our loss function we used mean absolute error.  We fit the model using fit_generator to train with the data in batches.  We created generators for training and validations and ran the model for 30, 1000-step epochs. 

We plot our training and validation loss to ensure we are not overfitting the data.

Finally we load our test data, create the feature matrix, and write our prediction to the submission file. 