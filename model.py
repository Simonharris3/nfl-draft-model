import copy
import re
import numpy as np
import tensorflow as tf
import keras
from keras import initializers, backend
import csv
import random
import math

num_inputs = 38
epochs = 50
neurons = [14, 12, 10, 8, 6, 4, 2, 1]
lr = 0.001
he_normal = True
position_dict = {'QB': 0.0, 'OT': 1.0, 'OL': 2.0, 'OG': 2.0, 'C': 3.0, 'RB': 4.0, 'HB': 4.0, 'FB': 4.0, 'TE': 5.0,
                 'WR': 6.0, 'DT': 7.0, 'DL': 8.0, 'DE': 9.0, 'EDGE': 9.0, 'OLB': 10.0, 'LB': 11.0, 'CB': 12.0,
                 'DB': 13.0, 'S': 14.0, 'P': 15.0, 'K': 16.0, 'LS': 17.0}


def main():
    base_file = open("sportsref_download_with_pff.csv", mode='r')
    # base_file = open("sportsref_download_with_pff.csv", mode='r')
    # def_files = []
    # for i in range(5):
    #     def_files.append(open("defense_summary_201" + str(i+6) + ".csv", mode='r'))
    # for i in range(5):
    #     reader = csv.reader(def_files[i])
    #     next(reader)
    # ol_files = []
    # for i in range(5):
    #     ol_files.append(open("offense_blocking_201" + str(i+6) + ".csv", mode='r'))
    # rb_files = []
    # for i in range(5):
    #     rb_files.append(open("rushing_summary_201" + str(i+6) + ".csv", mode='r'))
    # wr_files = []
    # for i in range(5):
    #     wr_files.append(open("receiving_summary_201" + str(i+6) + ".csv", mode='r'))

    reader = csv.reader(base_file)
    next(reader)

    # contains all the data in both training and test sets
    inputs_outputs = []

    for row in reader:
        inputs_outputs.append([])
        # don't count the first column (player name) or the last column (nfl pff grade which currently isn't being used)
        for i in range(len(row[1:-1])):
            value = row[i + 1]
            # for the first value (position), set this value to true so
            # the preprocess function knows to one-hot encode the data
            to_categorical = i == 0
            # the combine data is everything before the 9th value
            is_combine = i < 9
            # preprocess the data point, and tack it onto the end of the list
            # (which includes all the data for the current player)
            inputs_outputs[-1] += preprocess(value, to_categorical, is_combine)

    # for i in range(len(inputs_outputs)):
    #     if len(inputs_outputs[i]) != num_inputs+1:
    #         print(len(inputs_outputs[i]))
    #         raise ValueError("Expected length of " + str(num_inputs+1) + ", got: " + str(inputs_outputs[i]))

    train_input, train_output, test_input, test_output = split_train_test(inputs_outputs)

    model = keras.models.load_model("model")
    # layers = [layer for layer in keras.models.load_model("model").layers]

    # if he_normal:
    #     initializer = keras.initializers.HeNormal()
    # else:
    #     initializer = keras.initializers.GlorotUniform()

    # layers = []
    # # the neurons list maps out how many neurons should be in each layer
    # for i in range(len(neurons)):
    #     layers.append(keras.layers.Dense(neurons[i], activation="relu", kernel_initializer=initializer))
    #     layers.append(keras.layers.LeakyReLU())

    # layers.append(keras.layers.Dense(1, activation="relu",
    #                                  kernel_initializer=initializer, name="new_dense"))
    # layers.insert(1, keras.layers.LeakyReLU(name="new_leaky"))

    # model = keras.models.Sequential(layers)

    model.build((None, num_inputs))
    model.summary()

    # lrate_scheduler = keras.callbacks.LearningRateScheduler(step_decay)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=lr))
    model.fit(train_input, train_output, epochs=epochs)
    score = model.evaluate(test_input, test_output)

    if score < 61:
        model.save("model")
        print("model saved!")

    richardson_data = np.array([one_hot(0) + [76, 244, 4.43, 40.5, 0, 129, 0, 0, 767, 80.3, 192,
                                              74.8, 14, 65.5, 0, 0, 0, 0, 0, 0]])
    richardson_prediction = model.predict(richardson_data)[0][0]
    bad_qb_data = np.array([one_hot(0) + [67, 150, 4.9, 32, 14, 105, 7.7, 4.8, 767, 30, 900, 35, 900,
                                          40, 0, 0, 0, 0, 0, 0]])
    good_qb_data = np.array(
        [one_hot(0) + [78, 254, 4.33, 41, 20, 131, 6.8, 3.9, 800, 90.3, 700,
                       87.2, 900, 97.6, 800, 88.5, 0, 0, 0, 0]])
    bad_qb_prediction = model.predict(bad_qb_data)[0][0]
    good_qb_prediction = model.predict(good_qb_data)[0][0]
    branch_data = np.array([one_hot(position_dict['S']) + [72, 190, 4.58, 34.5, 14, 125, 0, 0, 768, 89.5, 624, 76.6, 290,
                                          72.4, 0, 0, 0, 0, 0, 0]])
    branch_prediction = model.predict(branch_data)[0][0]
    higher_cfs, lower_cfs = counterfactuals(model, richardson_data)
    print("Anthony Richardson prediction: pick " + str(richardson_prediction))
    print("bad qb prediction: pick " + str(bad_qb_prediction))
    print("good qb prediction: pick " + str(good_qb_prediction))
    print("Brian Branch prediction: pick " + str(branch_prediction) + "\n")

    lower_cfs.sort(key=lambda x: x[1])  # sort by the model's prediction
    higher_cfs.sort(key=lambda x: x[1])
    # worse lower richardson, worse higher richardson, etc
    wlr = []
    whr = []
    blr = []
    bhr = []

    for i in range(len(lower_cfs)):
        if lower_cfs[i][1] <= richardson_prediction:
            blr.append(lower_cfs[i])
        else:
            wlr.append(lower_cfs[i])

    for i in range(len(higher_cfs)):
        if higher_cfs[i][1] <= richardson_prediction:
            bhr.append(higher_cfs[i])
        else:
            whr.append(higher_cfs[i])

    # take only the most extreme changes in richardson's pick number
    wlr = wlr[-3:]
    whr = whr[-3:]
    blr = blr[:3]
    bhr = bhr[:3]
    if len(wlr) == 0 and len(whr) == 0:
        print("richardson can't get any worse!")
    else:
        print("richardson would be worse if the following attributes were higher: ")
        for i in range(len(whr)):
            print("%s (pick %.1f)" % (whr[i][0], whr[i][1]))
        print("\nrichardson would be worse if the following attributes were lower: ")
        for i in range(len(wlr)):
            print("%s (pick %.1f)" % (wlr[i][0], wlr[i][1]))

    if len(blr) == 0 and len(bhr) == 0:
        print("\nrichardson can't get any better!")
    else:
        print("\nrichardson would be better if the following attributes were higher: ")
        for i in range(len(bhr)):
            print("%s (pick %.1f)" % (bhr[i][0], bhr[i][1]))
        print("\nrichardson would be better if the following attributes were lower: ")
        for i in range(len(blr)):
            print("%s (pick %.1f)" % (blr[i][0], blr[i][1]))


def preprocess(value, to_categorical, is_combine):
    if value == '':
        rvalue = 0.0
    elif is_num(value):
        rvalue = float(value)
    elif '/' in value:
        # this is the cell with the pick information in it; we need to extract just the overall number of the pick
        # split the string into 4 parts: team / round / pick num / year
        pick_num = value.split(" / ")
        # find the first letter in the text (should be "rd" or "st" or "nd" or "th")
        ind = re.search("[a-zA-Z]", pick_num[2]).span(0)[0]

        # assert (pick_num[2][3:5] == "rd" or pick_num[2][3:5] == "st" or pick_num[2][3:5] == "nd" or
        #         pick_num[2][3:5] == "th", "value: " + pick_num[2][3:4])

        # the overall pick number is in the 3rd part of the string, and we cut out the text after the number
        rvalue = float(pick_num[2][:ind])
    else:
        try:
            rvalue = position_dict[value]
        except KeyError:
            raise Exception("Unknown input: %s" % value)

    if to_categorical:
        rvalue = one_hot(rvalue)
    # elif is_combine:
    #     if rvalue == 0:
    #         # for the combine data, we insert a data point telling the model whether the player participated in the
    #         # combine drill (to potentially improve performance)
    #         rvalue = [0, rvalue]
    #     else:
    #         rvalue = [1, rvalue]
    else:
        rvalue = [rvalue]

    # if type(rvalue) == np.ndarray:
    #     raise Exception("unexpected numpy array: " + str(rvalue))

    return rvalue


def split_train_test(data):
    random.shuffle(data)

    num_train_data = int(.8 * len(data))  # number of elements that will be in the training data set
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
    num_classes = 18

    positive_cfs = []
    negative_cfs = []

    i = num_classes + 1
    while i < original.size:
        modified = copy.deepcopy(original)
        modified[0, i] *= 1.05
        positive_cfs.append((cf_datum(i - num_classes), model.predict(np.array(modified))[0][0]))
        modified[0, i] *= .85
        negative_cfs.append((cf_datum(i - num_classes), model.predict(np.array(modified))[0][0]))

        i += 1

    return positive_cfs, negative_cfs


def cf_datum(i):
    if i == 1:
        return "height"
    if i == 2:
        return "weight"
    if i == 3:
        return "40 time"
    if i == 4:
        return "vert"
    if i == 5:
        return "bench"
    if i == 6:
        return "broad"
    if i == 7:
        return "3cone"
    if i == 8:
        return "shuttle"
    if i == 9:
        return "2021 snaps"
    if i == 10:
        return "2021 grade"
    if i == 11:
        return "2020 snaps"
    if i == 12:
        return "2020 grade"
    if i == 13:
        return "2019 snaps"
    if i == 14:
        return "2019 grade"
    if i == 15:
        return "2018 snaps"
    if i == 16:
        return "2018 grade"
    if i == 17:
        return "2017 snaps"
    if i == 18:
        return "2017 grade"
    if i == 19:
        return "2016 snaps"
    if i == 20:
        return "2016 grade"
    return "idek man"


def one_hot(n):
    return keras.utils.to_categorical(n, num_classes=18).tolist()


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
