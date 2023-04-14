import numpy as np
import tensorflow as tf
import keras
import csv
import random

num_inputs = 9


def main():
    file = open("sportsref_download.csv", mode='r')
    reader = csv.reader(file)
    next(reader)

    inputs_outputs = []

    for row in reader:
        inputs_outputs.append([])
        for value in row[1:]:
            num = value
            if value == '':
                num = '0'
            if value == 'QB':
                num = '0'
            if value == 'OT':
                num = '1'
            if value == 'OL':
                num = '2'
            if value == 'OG':
                num = '3'
            if value == 'C':
                num = '4'
            inputs_outputs[-1].append(float(num))

    random.shuffle(inputs_outputs)

    num_train_data = int(.8 * len(inputs_outputs))  # number of elements that will be in the training data set
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

    input_layer = keras.Input(shape=num_inputs)
    l1 = keras.layers.Dense(5, activation="relu")
    l2 = keras.layers.Dense(5, activation="relu")
    output = keras.layers.Dense(1, activation="relu")
    model = keras.models.Sequential([input_layer, l1, l2, output])
    model.summary()

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=.005))
    model.fit(train_input, train_output, batch_size=7, epochs=2200)
    model.evaluate(test_input, test_output)
    richardson = model.predict(np.array([[0, 76, 244, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]

    negative_counterfactuals = [("position", model.predict(np.array([[3, 65, 244, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]),
                                ("height", model.predict(np.array([[0, 65, 244, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]),
                                ("weight", model.predict(np.array([[0, 76, 230, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]),
                                ("40 time", model.predict(np.array([[0, 76, 244, 4.63, 40.5, 0, 129, 0, 0]]))[0][0]),
                                ("vert", model.predict(np.array([[0, 76, 244, 4.43, 36.5, 0, 129, 0, 0]]))[0][0]),
                                ("broad", model.predict(np.array([[0, 76, 244, 4.43, 40.5, 0, 110, 0, 0]]))[0][0])]

    positive_counterfactuals = [("height", model.predict(np.array([[0, 84, 244, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]),
                                ("weight", model.predict(np.array([[0, 76, 270, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]),
                                ("40 time", model.predict(np.array([[0, 76, 244, 4.23, 40.5, 0, 129, 0, 0]]))[0][0]),
                                ("vert", model.predict(np.array([[0, 76, 244, 4.43, 44.5, 0, 129, 0, 0]]))[0][0]),
                                ("bench", model.predict(np.array([[0, 76, 244, 4.43, 40.5, 30, 129, 0, 0]]))[0][0]),
                                ("broad", model.predict(np.array([[0, 76, 244, 4.43, 40.5, 0, 147, 0, 0]]))[0][0]),
                                ("3cone", model.predict(np.array([[0, 76, 244, 4.43, 40.5, 0, 129, 6.7, 0]]))[0][0]),
                                ("shuttle",
                                 model.predict(np.array([[0, 76, 244, 4.43, 40.5, 0, 129, 6.7, 4.29]]))[0][0])]

    print("Anthony Richardson prediction: pick " + str(round(richardson)))

    negative_counterfactuals.sort(key=lambda x: x[1])  # sort by the model's prediction
    positive_counterfactuals.sort(key=lambda x: x[1])
    worse_richardson = negative_counterfactuals[-3:]
    better_richardson = positive_counterfactuals[:3]

    i = 0
    while i < len(worse_richardson):
        if worse_richardson[i][1] <= richardson:
            worse_richardson.pop(i)
            i -= 1
        i += 1

    while i < len(better_richardson):
        if better_richardson[i][1] >= richardson:
            better_richardson.pop(i)
            i -= 1
        i += 1

    wr = worse_richardson
    if len(wr) == 0:
        print("richardson can't get any worse!")
    if len(wr) == 1:
        print("richardson would only be worse if his %s was worse (pick %.2f)." % (wr[0][0], wr[0][1]))
    if len(wr) == 2:
        print("richardson would be worse if his %s (pick %.2f) or his %s (pick %.2f) was worse." %
              (wr[0][0], wr[0][1], wr[1][0], wr[1][1]))
    if len(wr) == 3:
        print("richardson would be worse if his %s (pick %.2f), his %s (pick %.2f), or his %s (pick %.2f) was worse." %
              (wr[0][1], wr[0][1], wr[1][0], wr[1][1], wr[2][0], wr[2][1]))

    br = better_richardson
    if len(br) == 0:
        print("richardson can't get any better!")
    if len(br) == 1:
        print("richardson would only be better if his %s was better (pick %.2f)." % (br[0][0], br[0][1]))
    if len(br) == 2:
        print("richardson would be better if his %s (pick %.2f) or his %s (pick %.2f) was better." %
              (br[0][0], br[0][1], br[1][0], br[1][1]))
    if len(br) == 3:
        print("richardson would be better if his %s (pick %.2f), his %s (pick %.2f), or his %s (pick %.2f) was better."
              % (br[0][1], br[0][1], br[1][0], br[1][1], br[2][0], br[2][1]))

    # check if there's a way we could make one of richardson's measurables worse to make the model
    # like him better, or vice versa
    if negative_counterfactuals[0][1] < richardson:
        # strange_better_richardson is too long
        sbr = negative_counterfactuals[0]
        print("strangely, richardson would be better if his %s was worse (pick %.2f)." % (sbr[0], sbr[1]))

    if positive_counterfactuals[0][1] > richardson:
        swr = positive_counterfactuals[-1]
        print("strangely, richardson would be worse if his %s was better (pick %.2f)." % (swr[0], swr[1]))


main()
