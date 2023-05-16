import keras
from keras import Input, losses, optimizers
from keras.layers import Lambda, Dense, Concatenate

input_tensor = Input(shape=(5,))
group1 = Lambda(lambda x: x[0:2], output_shape=(2,))(input_tensor)
group2 = Lambda(lambda x: x[2:4], output_shape=(2,))(input_tensor)
group3 = Lambda(lambda x: x[4:], output_shape=(2,))(input_tensor)

l2_1 = Dense(1)(group1)
l2_2 = Dense(1)(group2)
l2_3 = Dense(1)(group3)

full_l2 = Concatenate()([l2_1, l2_2, l2_3])
output_tensor = Dense(2)(full_l2)

model = keras.models.Model(input_tensor, output_tensor)
model.build((5,))
model.summary()
x_train = [[1, 2, 3, 4, 5], [4, 4, 7, 8, 1], [8, 1, 2, 1, 3]]
y_train = [10, 12.5, 9]

x_test = [2, 5, 6, 4, 1]
y_test = [9.5]

model.compile(loss=losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=0.001))
model.fit(x_train, y_train, epochs=30)
score = model.evaluate(x_test, y_test)