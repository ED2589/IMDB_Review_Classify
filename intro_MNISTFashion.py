## FASHION MNIST Data

# Step 1 : loading and looking at data
# https://techwithtim.net/tutorials/python-neural-networks/loading-data/

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Step 1a: load MNIST Fashion data (images of diff types of clothing)
# 60 K train images, 10 K test images
data = keras.datasets.fashion_mnist
# Step 1b: keras way of splitting test and train data (x) /labels (y)
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Step 1c: types of clothing to classify - index = labels ranging from 0 to 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(train_labels[1:10])
# preview an image from dataset
# Step 1d: preview data
plt.imshow(train_images[7], cmap = plt.cm.binary)
#plt.show()
# print pixels in a single train image (28 * 28 pixels, each a number aka RGB value ranging from 0 to 255)
    #print(train_images[7])

# Step 1e: scale down the data - by dividing each train & test image (consisting of 28 by 28 RGB values ranging from 0 to 255)
#                       by 255
# so that easier to work with model
train_images = train_images/255
test_images = test_images/255

# print(train_images[7])

## Step 2 : creating a model
# https://techwithtim.net/tutorials/python-neural-networks/creating-a-model/

# flatten the data
#   - convert each image, which consists of 28 elements, with
#   each element being a list of 28 RGB values, to 784 RGB values in 1 list,
#    since input layer neurons only take FLATTENED list
#   - input layer in neural network: 784 neurons, each rep. a pixel value (between 0 and 1)
#   - output layer in neural network: 10 neurons, corresponding to each of the 10 classes
#                                       each containing a probablity (how likely it is the test image belongs to this class)
#   -hidden layer: has 128 neurons - allows more complex neural network (more weights and biases) - more accurate classification

#Step 2a: model architecture

#keras sequential API create neural nets layer-by-layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # flat input layer
    keras.layers. Dense(128, activation='relu'), # dense hidden layer, fully connected
                                                # activation function = rectified layer unit
    keras.layers.Dense(10,activation='softmax') # dense output layer, fully connected
                                                # activation function = softmax
    ])

# model params
# 'adam' = a stochastic gradient descent algorithm
# "sparse_categorical_entropy" = loss function  to minimize, used for multiclass labels (here have 10 classes)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# Step 2b: fit the model
# epochs = # of times neural net model will see the SAME training data (aka the same image)
# epochs (may or may not) affect accuracy of model since order and frequency by which neural net sees each training image matters
model.fit(train_images, train_labels, epochs = 5)

# Step 2c: test model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested accuracy: ", test_acc) # expected: ~0;87

## Step 3 : make a prediction using the model
# https://techwithtim.net/tutorials/python-neural-networks/making-predictions/

#Step 3a: make a prediction
#which clothing item is it?
# 1st argument of ``model.predict()` takes in a list, even if for a single test input
prediction = model.predict(test_images)
#print(prediction) # will get list of 10K arrays, each of length 10
                    # interpretation: 10 neurons in output layer, each rep. probability of each of 10 classes predicted
                    # 10K test images in total so 10K arrays generated

# since softmax activation function used, print MAX of the elements in each array
# to get the class the test input is most likely to be in
# `np.argmax` prints the index of the max. element - aka the output neuron #
#  that has highest prob.

#for i in range(len(prediction)):
    #print(np.argmax(prediction[i]))

# to get which class name the prediction is classified as, pass as index
# into class names in Line 19
# for i in range(len(prediction)):
#     print(class_names[np.argmax(prediction[i])])

# Step 3b: manually check predicted output against input for select few test images
# use matplotlib to show this for, say, 5 images
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap = plt.cm.binary) # show actual test image in plot
    plt.xlabel("Actual: " + class_names[test_labels[i]]) # show test input class on x-axis
    plt.title("predicted by NN: " + class_names[np.argmax(prediction[i])] ) # show predicted class as title

    plt.show()
