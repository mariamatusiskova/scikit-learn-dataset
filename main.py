from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf


def iris():
    # load the iris dataset
    iris_database = datasets.load_iris()

    # show data
    print('\nDataset (SepalLengthCm, SepalWidthCm, PetalLengthCM, PetalWidthCm): ')
    print(iris_database.data[:5])
    print('\nLabels: ')
    print(iris_database.target[:5])

    # separate features (x) and target variable (y)
    x = iris_database.data
    t_y = iris_database.target

    show_database_plot(iris_database)

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
    # train_test_split - utility for splitting a dataset into random train and test sets
    # 10% of data for testing, 90% for training
    x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.10)
    x_training, x_val, y_training, y_val = train_test_split(x_training, y_training, test_size=0.1)
    # print(f'x_training: \n{x_training}\n x_test: \n{x_test}\n x_val: \n{x_val}\n y_training: \n{y_training}\n y_test: \n{y_test}\n y_val: \n{y_val}\n')
    # after two splits, the program has three sets of data
    # - training set 80%
    # - validation set 10% (the validation helps ensure that model is learning well in some new situations)
    # - test set 10%

    ### build models ###

    # model one --------------------------------------------------------------------------------------------------------
    model_one = Sequential()

    # input layer
    model_one.add(Dense(16, input_shape=(4,), activation='relu', name='input_layer'))
    # hidden layers
    model_one.add(Dense(32, activation='relu', name='hidden_layer1'))
    model_one.add(Dense(16, activation='relu', name='hidden_layer2'))
    # output layer
    model_one.add(Dense(3, activation='softmax', name='output'))

    model_compile_summary_fit(model_one, x_val, y_val, x_training, y_training, x_test, y_test, f'{model_one=}'.split('=')[0])

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

    model_compile_summary_fit(model_two, x_val, y_val, x_training, y_training, x_test, y_test, f'{model_two=}'.split('=')[0])

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

    model_compile_summary_fit(model_three, x_val, y_val, x_training, y_training, x_test, y_test, f'{model_three=}'.split('=')[0])

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

    model_compile_summary_fit(model_four, x_val, y_val, x_training, y_training, x_test, y_test, f'{model_four=}'.split('=')[0])


def show_database_plot(iris_database):

    # Create a 2D scatter plot
    colorful_labels = plt.FuncFormatter(lambda i, *args: iris_database.target_names[int(i)])

    plt.scatter(iris_database.data[:, 0], iris_database.data[:, 1], c=iris_database.target)
    plt.colorbar(ticks=[0, 1, 2], format=colorful_labels)
    plt.xlabel(iris_database.feature_names[0])
    plt.ylabel(iris_database.feature_names[1])
    # edits layout of the plot to prevent clipping of labels or other elements
    plt.tight_layout()

    # Show the plot
    plt.show()


def model_compile_summary_fit(model, x_val, y_val, x_training, y_training, x_test, y_test, model_name):
    optimizer = Adam(learning_rate=0.001)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    model.summary()

    # train model (the neural network)
    history_results = model.fit(
        x_training,
        y_training,
        verbose=2,
        epochs=50,
        validation_data=(x_val, y_val)
    )

    create_accuracy_graph(model_name, history_results)
    create_loss_graph(model_name, history_results)

    # test
    test(model, x_test, y_test, x_training, y_training)


def create_loss_graph(model_name, history_results):
    # validation accuracy  is used to assess how well the model generalizes to new, unseen data
    plt.plot(range(50), history_results.history['loss'], color='b', label='Training loss')
    plt.plot(range(50), history_results.history['val_loss'], color='r', label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title(f'Loss over Epochs of the {model_name}')
    plt.legend()
    plt.show()


def create_accuracy_graph(model_name, history_results):
    # validation loss  is used to assess how well the model generalizes to new, unseen data
    plt.plot(range(50), history_results.history['accuracy'], color='g', label='Training accuracy')
    plt.plot(range(50), history_results.history['val_accuracy'], color='orange', label='Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title(f'Accuracy over Epochs of the {model_name}')
    plt.legend()
    plt.show()


def test(model, x_test, y_test, x_training, y_training):
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'--------- Test Loss: {round(test_loss * 100, 4)}%\n--------- Test Accuracy: {round(test_accuracy * 100, 4)}%')

    print("--------- Training predictions:")
    test_training = model.predict(x_training)
    training_positives = tf.argmax(test_training, axis=1)

    true_positives = tf.argmax(y_training, axis=1)
    print(metrics.classification_report(true_positives, training_positives, digits=3))
    print(metrics.confusion_matrix(true_positives, training_positives))

    print("--------- Test predictions:")
    test_prediction = model.predict(x_test)
    predicted_positives = tf.argmax(test_prediction, axis=1)

    true_positives = tf.argmax(y_test, axis=1)
    print(metrics.classification_report(true_positives, predicted_positives, digits=3))
    print(metrics.confusion_matrix(true_positives, predicted_positives))


if __name__ == '__main__':
    iris()
