import numpy as np
import tensorflow as tf
import keras
import numpy
import csv
import random

input_shape = 10


def main():
    file = open("draft data for fun.csv", mode='r')
    reader = csv.reader(file)
    next(reader)

    inputs_outputs = []

    for row in reader:
        inputs_outputs.append([])
        for value in row[1:]:
            inputs_outputs[-1].append(float(value))

    random.shuffle(inputs_outputs)

    num_train_data = int(.9 * len(inputs_outputs))  # number of elements that will be in the training data set
    # first n data points are the training set, rest are the test set
    train_set = inputs_outputs[:num_train_data]
    train_input = []
    train_output = []
    for player in train_set:
        # first num_classes-1 values are the input, last value is the output
        train_input.append(player[:-1])
        train_output.append(player[-1])

    test_set = inputs_outputs[num_train_data:]
    test_input = []
    test_output = []
    for player in test_set:
        # first num_classes-1 values are the input, last value is the output
        test_input.append(player[:-1])
        test_output.append(player[-1])

    input_layer = keras.Input(shape=input_shape)
    l1 = keras.layers.Dense(3, activation="sigmoid")
    output = keras.layers.Dense(1, activation="sigmoid")
    model = keras.models.Sequential([input_layer, l1, output])
    model.summary()

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=5))
    model.fit(train_input, train_output, batch_size=10, epochs=300)
    print(model.predict(np.array([[11, 8.3, 73, 195, 4.44, 38.5, 122, 6.98, 4.19, 30.625]])))
    print(model.get_weights())
    score = model.evaluate(test_input, test_output)


main()
