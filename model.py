import copy
import re
import sys

import numpy as np
import tensorflow as tf
import keras
from keras import initializers
import csv
import random
import math

num_inputs = 40
epochs = 40
neurons = [15, 15, 15, 1]
lr = 0.0003
train_percentage = .8

he_normal = True
position_dict = {'QB': 0.0, 'OT': 1.0, 'OL': 2.0, 'OG': 2.0, 'C': 3.0, 'RB': 4.0, 'HB': 4.0, 'FB': 4.0, 'TE': 5.0,
                 'WR': 6.0, 'DT': 7.0, 'DL': 8.0, 'DE': 9.0, 'EDGE': 9.0, 'OLB': 10.0, 'LB': 11.0, 'CB': 12.0,
                 'DB': 13.0, 'S': 14.0, 'P': 15.0, 'K': 16.0, 'LS': 17.0}
num_positions = 18
num_combine_data = 8
row_length = 21
min_snaps = 140

load_model = False
save_model_threshold = 52

cfs_threshold = 8


def main():
    base_file = open("sportsref_download_with_pff.csv", mode='r')

    reader = csv.reader(base_file)
    next(reader)

    # contains all the data in both training and test sets
    inputs_outputs = []

    for row in reader:
        inputs_outputs.append(preprocess_row(row))
        # don't count the first column (player name) or the last column (nfl pff grade which currently isn't being used)

    train_input, train_output, test_input, test_output = split_train_test(inputs_outputs)

    if load_model:
        model = keras.models.load_model("model")

    else:
        if he_normal:
            initializer = initializers.HeNormal()
        else:
            initializer = initializers.GlorotUniform()

        layers = []
        # the neurons list maps out how many neurons should be in each layer
        for i in range(len(neurons)):
            layers.append(keras.layers.Dense(neurons[i], activation="relu", kernel_initializer=initializer))
            layers.append(keras.layers.LeakyReLU())

        model = keras.models.Sequential(layers)

    model.build((None, num_inputs))
    model.summary()

    # lrate_scheduler = keras.callbacks.LearningRateScheduler(step_decay)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=lr))
    model.fit(train_input, train_output, epochs=epochs)
    score = model.evaluate(test_input, test_output)

    if score < save_model_threshold:
        model.save("model")
        print("model saved!")

    # test_cases = []

    richardson_data = ['richardson', 'QB', 73, 98, 1, 100, -50, 100, -50, -50,
                       767, 80.3, 192, 74.8, 14, 65.5, 0, 0, 0, 0, 0, 0, 81]
    # test_cases.append(preprocess_row(richardson_data))

    bad_qb_data = ['bad qb', 'QB', 7, 3, 97, 4, 12, 9, 87, 92,
                   0, 0, 900, 25, 900, 30, 900, 35, 0, 0, 0, 0, 81]
    # test_cases.append(preprocess_row(bad_qb_data))

    good_qb_data = ['good qb', 'QB', 96, 100, 6, 100, 98, 99, 0, 2,
                    0, 0, 700, 95.2, 900, 92.6, 800, 50, 800, 73, 0, 0, 81]
    # test_cases.append(preprocess_row(good_qb_data))

    branch_data = ['branch', 'S', 40, 12, 56, 35, 24, 77, -50, -50,
                   768, 89.5, 624, 76.6, 290, 72.4, 0, 0, 0, 0, 0, 0, 81]
    # test_cases.append(preprocess_row(branch_data))

    lawrence_data = ['lawrence', 'QB', 95, 44, -50, -50, -50, -50, -50, -50,
                     0, 0, 624, 91.1, 841, 91.1, 776, 90.7, 0, 0, 0, 0, 81]

    mid_ed_data = ['mid edge', 'EDGE', 63, 44, 31, 62, 85, 21, 45, 56,
                   500, 74, 624, 60, 841, 83, 776, 50, 0, 0, 0, 0, 81]

    model_darling_data = ['model darling', 'QB', 1, 1, 1, 1, 50, 12, 1, 1,
                   500, 1, 624, 1, 841, 99, 776, 1, 500, 1, 500, 99, 81]

    # test_cases.append(preprocess_row(lawrence_data))
    #
    # predictions = model.predict(np.asarray(test_cases), verbose=0)
    # richardson_prediction = predictions[0][0]
    #
    # print("Anthony Richardson prediction: pick " + str(richardson_prediction))
    # print("bad qb prediction: pick " + str(predictions[1][0]))
    # print("good qb prediction: pick " + str(predictions[2][0]))
    # print("Brian Branch prediction: pick " + str(predictions[3][0]))
    # print("Trevor Lawrence prediction: pick " + str(predictions[4][0]) + '\n')

    print()
    # run_counterfactuals(model, richardson_data)
    run_counterfactuals(model, bad_qb_data)
    run_counterfactuals(model, good_qb_data)
    # run_counterfactuals(model, branch_data)
    # run_counterfactuals(model, lawrence_data)
    # run_counterfactuals(model, mid_ed_data)
    run_counterfactuals(model, model_darling_data)


def run_counterfactuals(model, data):
    player_name = data[0]
    higher_cfs, lower_cfs, no_data_cfs, optimals = counterfactuals(model, np.asarray([preprocess_row(data)]))
    prediction = round(model.predict(np.asarray([preprocess_row(data)]), verbose=0)[0][0], 1)
    print(player_name + " prediction: pick " + str(prediction))

    lower_cfs.sort(key=lambda x: x[1])  # sort by the model's prediction
    higher_cfs.sort(key=lambda x: x[1])
    no_data_cfs.sort(key=lambda x: x[1])

    # worse lower, better lower, etc
    wl_cfs = []
    bl_cfs = []

    for i in range(len(lower_cfs)):
        if lower_cfs[i][1] < prediction - cfs_threshold:
            bl_cfs.append(lower_cfs[i])
        elif lower_cfs[i][1] > prediction + cfs_threshold:
            wl_cfs.append(lower_cfs[i])

    wh_cfs = []
    bh_cfs = []

    for i in range(len(higher_cfs)):
        if higher_cfs[i][1] < prediction - cfs_threshold:
            bh_cfs.append(higher_cfs[i])
        elif higher_cfs[i][1] > prediction + cfs_threshold:
            wh_cfs.append(higher_cfs[i])

    wn_cfs = []
    bn_cfs = []

    for i in range(len(no_data_cfs)):
        if no_data_cfs[i][1] < prediction - cfs_threshold:
            bn_cfs.append(no_data_cfs[i])
        elif no_data_cfs[i][1] > prediction + cfs_threshold:
            wn_cfs.append(no_data_cfs[i])

    print("optimal stats for %s: " % player_name)
    for i in range(len(optimals)):
        print("%s: %.1f (pick %.1f)" % (optimals[i][0], optimals[i][1], optimals[i][2]))

    # if len(wh_cfs) != 0:
    #     print(player_name + " would be worse if the following attributes were higher: ")
    #     for i in range(len(wh_cfs)):
    #         print("%s (%.1f -> %.1f, pick %.1f)" % (wh_cfs[i][0], wh_cfs[i][2], wh_cfs[i][3], wh_cfs[i][1]))
    #
    # if len(wl_cfs) != 0:
    #     print("\n" + player_name + " would be worse if the following attributes were lower: ")
    #     for i in range(len(wl_cfs)):
    #         print("%s (%.1f -> %.1f, pick %.1f)" % (wl_cfs[i][0], wl_cfs[i][2], wl_cfs[i][3], wl_cfs[i][1]))

    # if len(wn_cfs) != 0:
    #     print("\n" + player_name + " would be worse if there was no data for the following attributes: ")
    #     for i in range(len(wn_cfs)):
    #         print("%s (pick %.1f)" % (wn_cfs[i][0], wn_cfs[i][1]))

    # if len(bh_cfs) != 0:
    #     print("\n" + player_name + " would be better if the following attributes were higher: ")
    #     for i in range(len(bh_cfs)):
    #         print("%s (%.1f -> %.1f, pick %.1f)" % (bh_cfs[i][0], bh_cfs[i][2], bh_cfs[i][3], bh_cfs[i][1]))

    # if len(bl_cfs) != 0:
    #     print("\n" + player_name + " would be better if the following attributes were lower: ")
    #     for i in range(len(bl_cfs)):
    #         print("%s (%.1f -> %.1f, pick %.1f)" % (bl_cfs[i][0], bl_cfs[i][2], bl_cfs[i][3], bl_cfs[i][1]))

    # if len(bn_cfs) != 0:
    #     print("\n" + player_name + " would be better if there was no data for the following attributes: ")
    #     for i in range(len(bn_cfs)):
    #         print("%s (pick %.1f)" % (bn_cfs[i][0], bn_cfs[i][1]))

    print("-----------------------------------------------------------------------")


# change the input to the model slightly to see how the result would change
def counterfactuals(model, original):
    higher_cfs = []
    lower_cfs = []
    no_data_cfs = []
    optimals = []

    original_prediction = model.predict(original, verbose=0)[0][0]
    i = num_positions + 1
    while i < original.size:
        is_combine = i < num_positions + num_combine_data * 2
        modified = copy.deepcopy(original)

        modified[0, i] += 10
        higher_cfs.append((cf_datum(i - num_positions), model.predict(modified, verbose=0)[0][0],
                           original[0, i], modified[0, i]))

        modified[0, i] -= 20
        lower_cfs.append((cf_datum(i - num_positions), model.predict(modified, verbose=0)[0][0],
                          original[0, i], modified[0, i]))

        optimal = (sys.maxsize, -81)
        k = 3
        while k <= 100:
            modified[0, i] = k
            prediction = model.predict(modified, verbose=0)[0][0]
            if prediction < optimal[0]:
                optimal = (prediction, k)

            k += 3

        if abs(optimal[0] - original_prediction) > cfs_threshold:
            optimals.append((cf_datum(i - num_positions), optimal[1], optimal[0]))

        if is_combine:
            modified[0, i - 1] = 0
            modified[0, i] = -50
        else:
            modified[0, i] = 0
        no_data_cfs.append((cf_datum(i - num_positions), model.predict(modified, verbose=0)[0][0]))

        if is_combine and not i == num_positions + num_combine_data * 2 - 1:
            i += 2
        else:
            i += 1

    return higher_cfs, lower_cfs, no_data_cfs, optimals


def preprocess_row(row):
    result = []
    for i in range(len(row[1:-1])):
        value = row[i + 1]
        # for the first value (position), set this value to true so
        # the preprocess function knows to one-hot encode the data
        # the combine data is everything before the 9th value
        is_combine = 0 < i <= num_combine_data
        # ignore the grades where the player didn't have enough snaps
        # after the combine data, the even column numbers contain the grades, and the odd ones contain the snaps
        if i > num_combine_data and i != row_length:
            try:
                ignore = float(row[i]) < min_snaps and i % 2 == 0
            except ValueError:
                ignore = True
        else:
            ignore = False

        if i <= num_combine_data or i % 2 == 0 or i == row_length:
            # preprocess the data point, and tack it onto the end of the list
            # (which includes all the data for the current player)
            result += preprocess(value, is_combine, ignore)

    return result


def preprocess(value, is_combine, ignore):
    if value == '':
        rvalue = [0.0]
    elif is_num(value):
        rvalue = [float(value)]
    elif '/' in value:
        # this is the cell with the pick information in it; we need to extract just the overall number of the pick
        # split the string into 4 parts: team / round / pick num / year
        pick_num = value.split(" / ")
        # find the first letter in the text (should be "rd" or "st" or "nd" or "th")
        ind = re.search("[a-zA-Z]", pick_num[2]).span(0)[0]

        # assert (pick_num[2][3:5] == "rd" or pick_num[2][3:5] == "st" or pick_num[2][3:5] == "nd" or
        #         pick_num[2][3:5] == "th", "value: " + pick_num[2][3:4])

        # the overall pick number is in the 3rd part of the string, and we cut out the text after the number
        rvalue = [float(pick_num[2][:ind])]
    else:
        try:
            rvalue = one_hot(position_dict[value])
        except KeyError:
            raise Exception("Unknown input: %s" % value)

    if is_combine:
        if rvalue[0] == -50:
            # for the combine data, we insert a data point telling the model whether the player participated in the
            # combine drill (to potentially improve performance)
            rvalue = [0, 0]
        else:
            rvalue.insert(0, 1.0)
    elif ignore:
        rvalue = [0.0]

    # if type(rvalue) == np.ndarray:
    #     raise Exception("unexpected numpy array: " + str(rvalue))

    return rvalue


def split_train_test(data):
    random.shuffle(data)

    num_train_data = int(train_percentage * len(data))  # number of elements that will be in the training data set
    # first n data points are the training set, rest are the test set
    train_set = data[:num_train_data]
    train_input = []
    train_output = []
    for player in train_set:
        # first num_classes-1 values are the input, last value is the output
        train_input.append(player[:-1])
        train_output.append(player[-1])

    test_set = data[num_train_data:]
    test_input = []
    test_output = []
    for player in test_set:
        # first num_classes-1 values are the input, last value is the output
        test_input.append(player[:-1])
        test_output.append(player[-1])

    # for i in range(len(test_input)):
    #     if len(test_input[i]) != 24:
    #         print(len(test_input[i]))
    #     if i<6:
    #         print(test_input[i])

    # convert to numpy arrays
    train_input = np.asarray(train_input)
    train_output = np.asarray(train_output)
    test_input = np.asarray(test_input)
    test_output = np.asarray(test_output)

    return train_input, train_output, test_input, test_output


def leaky_model():
    layers = [keras.Input(shape=num_inputs)]
    for i in range(len(neurons)):
        if he_normal:
            initializer = tf.keras.initializers.HeNormal()
        else:
            initializer = tf.keras.initializers.GlorotUniform()
        layers.append(keras.layers.Dense(neurons[i], activation="relu", kernel_initializer=initializer))
        layers.append(keras.layers.LeakyReLU())

    if he_normal:
        initializer = tf.keras.initializers.HeNormal()
    else:
        initializer = tf.keras.initializers.GlorotUniform()
    layers.append(keras.layers.Dense(1, activation="relu", kernel_initializer=initializer))
    return keras.models.Sequential(layers)


def cf_datum(i):
    if i == 1:
        return "height"
    if i == 3:
        return "weight"
    if i == 5:
        return "40 time"
    if i == 7:
        return "vert"
    if i == 9:
        return "bench"
    if i == 11:
        return "broad"
    if i == 13:
        return "3cone"
    if i == 15:
        return "shuttle"
    if i == 16:
        return "2021 grade"
    if i == 17:
        return "2020 grade"
    if i == 18:
        return "2019 grade"
    if i == 19:
        return "2018 grade"
    if i == 20:
        return "2017 grade"
    if i == 21:
        return "2016 grade"
    raise Exception("counterfactuals bug")


def one_hot(n):
    return keras.utils.to_categorical(n, num_classes=num_positions).tolist()


def step_decay(epoch):
    initial_lrate = 0.05
    drop = 0.8
    epochs_drop = (1.0 / math.log(0.00001 / initial_lrate, drop)) * epochs  # the learning rate should end at 0.00001
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


# return whether a string is numeric
def is_num(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


main()
