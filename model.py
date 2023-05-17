import copy
import re
import numpy as np
import tensorflow as tf
import keras
from keras import initializers
import csv
import random
import math

num_inputs = 40
epochs = 60
neurons = [14, 14, 12, 10, 8, 6, 4, 2, 1]
lr = 0.001
train_percentage = .8
he_normal = True
position_dict = {'QB': 0.0, 'OT': 1.0, 'OL': 2.0, 'OG': 2.0, 'C': 3.0, 'RB': 4.0, 'HB': 4.0, 'FB': 4.0, 'TE': 5.0,
                 'WR': 6.0, 'DT': 7.0, 'DL': 8.0, 'DE': 9.0, 'EDGE': 9.0, 'OLB': 10.0, 'LB': 11.0, 'CB': 12.0,
                 'DB': 13.0, 'S': 14.0, 'P': 15.0, 'K': 16.0, 'LS': 17.0}
num_positions = 18
num_combine_data = 8
row_length = 21
min_snaps = 100
load_model = False


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

    if score < 65:
        model.save("model")
        print("model saved!")

    test_cases = []

    richardson_data = ['richardson', 'QB', 76, 244, 4.43, 40.5, 0, 129, 0, 0,
                       767, 80.3, 192, 74.8, 14, 65.5, 0, 0, 0, 0, 0, 0, 81]
    test_cases.append(preprocess_row(richardson_data))

    bad_qb_data = ['bad', 'QB', 67, 150, 4.9, 32, 14, 105, 7.7, 4.8,
                   767, 30, 900, 35, 900, 40, 0, 0, 0, 0, 0, 0, 81]
    test_cases.append(preprocess_row(bad_qb_data))

    good_qb_data = ['good', 'QB', 78, 254, 4.33, 41, 20, 131, 6.8, 3.9,
                    800, 90.3, 700, 87.2, 900, 97.6, 800, 96.5, 0, 0, 0, 0, 81]
    test_cases.append(preprocess_row(good_qb_data))

    branch_data = ['branch', 'S', 72, 190, 4.58, 34.5, 14, 125, 0, 0, 768,
                   89.5, 624, 76.6, 290, 72.4, 0, 0, 0, 0, 0, 0, 81]
    test_cases.append(preprocess_row(branch_data))

    lawrence_data = ['lawrence', 'QB', 77, 213, 0, 0, 0, 0, 0, 0, 0, 0,
                     624, 91.1, 841, 91.1, 776, 90.7, 0, 0, 0, 0, 81]
    test_cases.append(preprocess_row(lawrence_data))

    predictions = model.predict(np.asarray(test_cases))
    richardson_prediction = predictions[0][0]

    print("Anthony Richardson prediction: pick " + str(richardson_prediction))
    print("bad qb prediction: pick " + str(predictions[1][0]))
    print("good qb prediction: pick " + str(predictions[2][0]))
    print("Brian Branch prediction: pick " + str(predictions[3][0]))
    print("Trevor Lawrence prediction: pick " + str(predictions[4][0]) + '\n')

    higher_cfs, lower_cfs, no_data_cfs = counterfactuals(model, np.asarray([preprocess_row(richardson_data)]))

    lower_cfs.sort(key=lambda x: x[1])  # sort by the model's prediction
    higher_cfs.sort(key=lambda x: x[1])
    no_data_cfs.sort(key=lambda x: x[1])
    # worse lower richardson, worse higher richardson, etc
    wlr = []
    blr = []

    for i in range(len(lower_cfs)):
        if lower_cfs[i][1] < richardson_prediction - 5:
            blr.append(lower_cfs[i])
        elif lower_cfs[i][1] > richardson_prediction + 5:
            wlr.append(lower_cfs[i])

    whr = []
    bhr = []

    for i in range(len(higher_cfs)):
        if higher_cfs[i][1] < richardson_prediction - 5:
            bhr.append(higher_cfs[i])
        elif higher_cfs[i][1] > richardson_prediction + 5:
            whr.append(higher_cfs[i])

    wnr = []
    bnr = []

    for i in range(len(no_data_cfs)):
        if no_data_cfs[i][1] < richardson_prediction - 5:
            wnr.append(no_data_cfs[i])
        elif no_data_cfs[i][1] > richardson_prediction + 5:
            bnr.append(no_data_cfs[i])

    # take only the most extreme changes in richardson's pick number
    wlr = wlr[-3:]
    whr = whr[-3:]
    blr = blr[:3]
    bhr = bhr[:3]
    wnr = wnr[-3:]
    bnr = bnr[:3]

    print("richardson would be worse if the following attributes were higher: ")
    for i in range(len(whr)):
        print("%s (pick %.1f)" % (whr[i][0], whr[i][1]))
    print("\nrichardson would be worse if the following attributes were lower: ")
    for i in range(len(wlr)):
        print("%s (pick %.1f)" % (wlr[i][0], wlr[i][1]))
    print("\nrichardson would be worse if there was no data for the following attributes: ")
    for i in range(len(wnr)):
        print("%s (pick %.1f)" % (wnr[i][0], wnr[i][1]))

    print("\nrichardson would be better if the following attributes were higher: ")
    for i in range(len(bhr)):
        print("%s (pick %.1f)" % (bhr[i][0], bhr[i][1]))
    print("\nrichardson would be better if the following attributes were lower: ")
    for i in range(len(blr)):
        print("%s (pick %.1f)" % (blr[i][0], blr[i][1]))
    print("\nrichardson would be better if there was no data for the following attributes: ")
    for i in range(len(bnr)):
        print("%s (pick %.1f)" % (bnr[i][0], bnr[i][1]))


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
        if rvalue[0] == 0:
            # for the combine data, we insert a data point telling the model whether the player participated in the
            # combine drill (to potentially improve performance)
            rvalue.insert(0, 0.0)
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


# change the input to the model slightly to see how the result would change
def counterfactuals(model, original):
    higher_cfs = []
    lower_cfs = []
    no_data_cfs = []

    i = num_positions + 1
    while i < original.size:
        modified = copy.deepcopy(original)
        modified[0, i] *= 1.05
        higher_cfs.append((cf_datum(i - num_positions), model.predict(modified)[0][0]))

        modified[0, i] *= .85
        lower_cfs.append((cf_datum(i - num_positions), model.predict(modified)[0][0]))

        modified[0, i] = 0
        no_data_cfs.append((cf_datum(i - num_positions), model.predict(modified)[0][0]))

        i += 2

    return higher_cfs, lower_cfs, no_data_cfs


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
    if i == 17:
        return "2021 grade"
    if i == 19:
        return "2020 grade"
    if i == 21:
        return "2019 grade"
    if i == 23:
        return "2018 grade"
    if i == 25:
        return "2017 grade"
    if i == 27:
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
