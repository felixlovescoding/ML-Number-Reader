# importing all the libraries
import os.path
import cv2
import keras.utils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# importing the keras number dataset
mnist = tf.keras.datasets.mnist

# declaring variables to load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize x_train and y_train
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

# flattens the image into 1D and sets its side to 28 by 28 (since it is a grey scale image)
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Add two dense layers each with 128 neurons and uses the Rectified Linear Unit (ReLU) activation function
# The ReLU function "activates" a neuron when its input is positive, and it remains inactive (outputting zero) when the input is negative.
# Increase or decrease the number of neurons based on your preference (for accuracy)
model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# sets the optimiser, goal(max accuracy or loss), and how to measure it
# 'adam' stands for "Adaptive Moment Estimation," and it combines the benefits of two other popular optimizers called RMSprop and Momentum.
# Sparse categorical crossentropy is a specific loss function used in deep learning, typically when the provided datasets are integers
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trains the model
# epochs is just the number of sets you are running (increase or decrease it based on your preference)
model.fit(x_train, y_train, epochs=5)

# Evaluates the training result of the model (you could compare it against the training model) Typically,
# we don't want the evaluation result to be too similar to the training results, this is to avoid "over-fit" or
# "local optima" where the model is learning for the sake of being good at a certain testing criteria rather than the overall picture
# For example, If we train a model to walk as fast as possible, the AI could end up crab walking or start moon-walking(moon-walking) rather than
# training to walk as we expected
# To avoid this problem, we could add in extra incentives / modify the existing incentives to reward / punish the model for staying on track
val_loss, val_accuracy = model.evaluate(x_test, y_test)

print(val_loss, val_accuracy)

# Shows the image in a 2D plot
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# Saves the trained model to your computer for future use
model.save('/Users/happy/AppData/Local/Deep Learning Model With Tensorflow/Number_Reader_Model1')

# Loads the saved model (I don't want to keep loading the models, so if you want to enable it just uncomment the code)
# new_model = tf.keras.models.load_model('/Users/happy/AppData/Local/Deep Learning model With Tensorflow/Number_Reader_Model1')

# image counter
image_number = 1
# checks if the files does exist at the give path
while os.path.isfile(f"/Users/happy/AppData/Local/Deep Learning Model Data/Digits/digit{image_number}.png"):
    try:
        # reads the image, but we only want the channels [height, width, channels]
        image = cv2.imread(f"/Users/happy/AppData/Local/Deep Learning Model Data/Digits/digit{image_number}.png")[:, :, 0]
        # inverts the image (dark area to light area and vice versa)
        image = np.invert(np.array([image]))

        # asks the model for predictions
        predictions = model.predict(image)
        # Prints out the humanly incomprehensible prediction made by the model
        print(f"This number is probably a {np.argmax(predictions)}")

        # Varies whether the prediction is correct or not by showing the actual 2D plot image of the given data
        plt.imshow(image[0], cmap=plt.cm.binary)
        plt.show()

    except:
        # in case of error
        print("error :(")

    finally:
        # counter increment
        image_number += 1
