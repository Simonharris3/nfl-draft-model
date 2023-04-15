import re

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
            inputs_outputs[-1].append(float(preprocess(value)))

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
    l1 = keras.layers.Dense(10, activation="relu")
    leaky1 = keras.layers.LeakyReLU()
    l2 = keras.layers.Dense(10, activation="relu")
    leaky2 = keras.layers.LeakyReLU()
    output = keras.layers.Dense(1, activation="relu")
    model = keras.models.Sequential([input_layer, l1, leaky1, l2, leaky2, output])
    model.summary()

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=.02))
    model.fit(train_input, train_output, batch_size=50, epochs=1000)
    model.evaluate(test_input, test_output)
    richardson = model.predict(np.array([[0, 76, 244, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]

    negative_counterfactuals = [("position", model.predict(np.array([[3, 76, 244, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]),
                                ("height", model.predict(np.array([[0, 71, 244, 4.43, 40.5, 0, 129, 0, 0]]))[0][0]),
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
                                ("3cone", model.predict(np.array([[0, 76, 244, 4.43, 40.5, 0, 129, 7.9, 0]]))[0][0]),
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
              (wr[0][0], wr[0][1], wr[1][0], wr[1][1], wr[2][0], wr[2][1]))

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
              % (br[0][0], br[0][1], br[1][0], br[1][1], br[2][0], br[2][1]))

    # check if there's a way we could make one of richardson's measurables worse to make the model
    # like him better, or vice versa
    if negative_counterfactuals[0][1] < richardson:
        # strange_better_richardson is too long
        sbr = negative_counterfactuals[0]
        print("strangely, richardson would be better if his %s was worse (pick %.2f)." % (sbr[0], sbr[1]))

    if positive_counterfactuals[0][1] > richardson:
        swr = positive_counterfactuals[-1]
        print("strangely, richardson would be worse if his %s was better (pick %.2f)." % (swr[0], swr[1]))


def preprocess(value):
    # create a position code for each position. change to one-hot
    if value == '':
        num = '0'
    elif value == 'QB':
        num = '0'
    elif value == 'OT':
        num = '1'
    elif value == 'OL' or value == 'OG':
        num = '2'
    elif value == 'C':
        num = '3'
    elif value == 'RB' or value == 'FB' or value == 'HB':
        num = '4'
    elif value == 'TE':
        num = '5'
    elif value == 'WR':
        num = '6'
    elif value == 'DT':
        num = '7'
    elif value == 'DL':
        num = '8'
    elif value == 'DE' or value == 'EDGE':
        num = '9'
    elif value == 'OLB':
        num = '10'
    elif value == 'LB':
        num = '11'
    elif value == 'CB':
        num = '12'
    elif value == 'DB':
        num = '13'
    elif value == 'S':
        num = '14'
    elif value == 'P':
        num = '15'
    elif value == 'K':
        num = '16'
    elif value == 'LS':
        num = '17'
    elif is_num(value):
        num = value
    elif '/' in value:
        pick_num = value.split(" / ")
        # find the first letter in the text (should be "rd" or "st" or "nd" or "th")
        ind = re.search("[a-zA-Z]", pick_num[2]).span(0)[0]

        assert (pick_num[2][3:5] == "rd" or pick_num[2][3:5] == "st" or pick_num[2][3:5] == "nd" or
                pick_num[2][3:5] == "th", "value: " + pick_num[2][3:4])

        num = pick_num[2][:ind]
    else:
        raise Exception("Unknown input: %s" % value)

    return num


def is_num(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


main()
