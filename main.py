from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# load the iris dataset
iris = datasets.load_iris()

# show data
print('\nDataset (SepalLengthCm, SepalWidthCm, PetalLengthCM, PetalWidthCm): ')
print(iris.data[:5])
print('\nLabels: ')
print(iris.target[:5])

# separate features (x) and target variable (y)
x = iris.data
t_y = iris.target

# uncomment the following lines if you want to print the feature and target arrays
# ## print(f'x: {x}')
# ## print(f'y: {y}')

# encoding target variable (species) to one-hot encoded arrays
# set sparse=False for a dense array, if True it is a matrix array
one_hot_encoder = OneHotEncoder(sparse_output=False)
y = one_hot_encoder.fit_transform(t_y.reshape(-1, 1))
print('\nTarget variables (One-Hot Encoded):')
print(y[:5])

# test, train and validation data split
# ## train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
# train_test_split - utility for splitting a dataset into random train and test sets
# 15% of data for testing, 85% for training
x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.15)
x_training, x_val, y_training, y_val = train_test_split(x_training, y_training, test_size=0.1)
# print(f'x_training: \n{x_training}\n x_test: \n{x_test}\n x_val: \n{x_val}\n y_training: \n{y_training}\n y_test: \n{y_test}\n y_val: \n{y_val}\n')
# after two splits, the program has three sets of data
# - training set 75%
# - validation set 10% (the validation helps ensure that model is learning well in some new situations)
# - test set 15%

### build models ###
optimizer = Adam(learning_rate=0.001)

# model one --------------------------------------------------------------------------------------------------------
model_one = Sequential()

# input layer
model_one.add(Dense(16, input_shape=(4,), activation='relu', name='input_layer'))
# hidden layers
model_one.add(Dense(32, activation='relu', name='hidden_layer1'))
model_one.add(Dense(16, activation='relu', name='hidden_layer2'))
# output layer
model_one.add(Dense(3, activation='softmax', name='output'))

# optimizer - weights
# loss - measures how well the model is performing
model_one.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

model_one.summary()

# train model (the neural network)
model_one.fit(
    x_training,
    y_training,
    verbose=2,
    epochs=50,
    validation_data=(x_val, y_val)
)

# test
test_loss, test_accuracy = model_one.evaluate(x_test, y_test)
print(f'Test Loss: {round(test_loss * 100, 4)}%, Test Accuracy: {round(test_accuracy * 100, 4)}%')

# model Two --------------------------------------------------------------------------------------------------------
model_two = Sequential()

# input layer
model_two.add(Dense(16, input_shape=(4,), activation='relu', name='input_layer'))
# hidden layers
# increased number of neurons in hidden_layer2
model_two.add(Dense(64, activation='relu', name='hidden_layer1'))
model_two.add(Dense(32, activation='relu', name='hidden_layer2'))
# added an additional hidden layer
model_two.add(Dense(16, activation='relu', name='hidden_layer3'))
# output layer
model_two.add(Dense(3, activation='softmax', name='output'))

model_two.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

model_two.summary()

# train model (the neural network)
model_two.fit(
    x_training,
    y_training,
    verbose=2,
    epochs=50,
    validation_data=(x_val, y_val)
)

# test
test_loss, test_accuracy = model_two.evaluate(x_test, y_test)
print(f'Test Loss: {round(test_loss * 100, 4)}%, Test Accuracy: {round(test_accuracy * 100, 4)}%')

# model three (an example with different architecture) -------------------------------------------------------------
model_three = Sequential()

# input layer
# increased the number of neurons in input_layer
model_three.add(Dense(16, input_shape=(4,), activation='relu', name='input_layer'))
# hidden layers
model_three.add(Dense(32, activation='relu', name='hidden_layer1'))
model_three.add(Dense(16, activation='relu', name='hidden_layer2'))
model_three.add(Dense(8, activation='relu', name='hidden_layer3'))
# output layer
model_three.add(Dense(3, activation='softmax', name='output'))

model_three.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

model_three.summary()

# train model (the neural network)
model_three.fit(
    x_training,
    y_training,
    verbose=2,
    epochs=50,
    validation_data=(x_val, y_val)
)

# test
test_loss, test_accuracy = model_three.evaluate(x_test, y_test)
print(f'Test Loss: {round(test_loss * 100, 4)}%, Test Accuracy: {round(test_accuracy * 100, 4)}%')

# Model Four -------------------------------------------------------------
model_four = Sequential()

# input layer
model_four.add(Dense(16, input_shape=(4,), activation='relu', name='input_layer'))
# hidden layers
model_four.add(Dense(32, activation='relu', name='hidden_layer1'))
model_four.add(Dense(16, activation='relu', name='hidden_layer2'))
# reduced the number of neurons in hidden_layer3
model_four.add(Dense(8, activation='relu', name='hidden_layer3'))
# output layer
model_four.add(Dense(3, activation='softmax', name='output'))

model_four.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

model_four.summary()

# train model (the neural network)
model_four.fit(
    x_training,
    y_training,
    verbose=2,
    epochs=50,
    validation_data=(x_val, y_val)
)

# test
test_loss, test_accuracy = model_four.evaluate(x_test, y_test)
# test accuracy overall correctness
print(f'Test Loss: {round(test_loss * 100, 4)}%, Test Accuracy: {round(test_accuracy * 100, 4)}%')

# Generate predictions using the test data
predictions = model_four.predict(x_test)
predicted_positives = tf.argmax(predictions, axis=1)

# Convert one-hot encoded labels to categorical labels
true_positives = tf.argmax(y_test, axis=1)

# Calculate precision and recall using scikit-learn functions
precision = precision_score(true_positives, predicted_positives, average='weighted')
recall = recall_score(true_positives, predicted_positives, average='weighted')

print(f'Test precision: {round(precision * 100, 4)}%')
print(f'Test recall: {round(recall * 100, 4)}%')

# mozem mat validation set?
