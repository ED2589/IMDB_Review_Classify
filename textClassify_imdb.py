'''
commented version of 'runthis.py'
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np # pip install numpy==1.16.1

## Step 1: load in imdb data
data = keras.datasets.imdb

## Step 2: split the data

# note: specify `num_words = 10000` to include the 10000 most frequently occurring words in
# the entire data set & omit the rest of the words (aka ones rarely used)
# since including them adds noise to data and makes it harder for model to accurately predict

# note: can also bump from 10,000 to more words (e.g. 88000) for better accuracy for prediction (see Step 5)
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 88000)

# print a movie review
# a list of integers (i.e. integer-encoded words)
# because canNOT pass strings to neural net
# print(train_data[9])

# preview length of some reviews
# for i in range(5):
# #     print (len(train_data[i]))

## Step 3: data preprocessing

# Step 3a: map integer-encodings for each movie review to actual words

# init. a dictionary w/ words (key) and int (val)
# usually user has to define own dictionary manually, but here word-to-val mapping provided by imdb dataset (how convenient)
word_index = data.get_word_index()
# print(type(word_index)) # dictionary data type
# print(word_index)
# print(word_index['branch'])
# print(word_index.items())

# Get a list of keys from dictionary which has the given value

# listOfKeys = [key  for (key, value) in word_index.items() if value == 3]
# print(listOfKeys)

# Step 3b: add 3 to `val` cuz will have 4 additional vals for non-valid chars in movie reviews
# note: not add 4 to 'val' cuz in original `word_index` NO keys have value = 0 (convention of data set)
word_index =  {key: (val+3) for key, val in word_index.items()}
# e.g. special char. 'PADDING' with value 0
#      since fixed number of neurons in model, so movie reviews should have same length
#      used later (ref: L81)
word_index["<PAD>"] = 0 # default start char # see doc 'start_char': https://keras.io/api/datasets/imdb/#get_word_index-function
word_index["<START>"] = 1 # all training/testing data starts (type list) starts with 1, so we encode as 'START' of review
word_index["<UNK>"] = 2 # took words out when loading data (L15) & default replacement in data is integer '2'  see doc 'oov_char': https://keras.io/api/datasets/imdb/#get_word_index-function
word_index["<UNUSED>"] = 3
#
# print(word_index['branch'])

# Step 3c: Tensorflow-defined word mapping: must do key = int, val = word
#   since movie review is rep. by list of int, which we have to map onto words
#   rather than the other way around
reverse_word_index = {val:key for key, val in word_index.items()}
# print(type(reverse_word_index2)) # dictionary data type
# print(reverse_word_index[9459]) # 'branch'

# Step 3d: decode train, test data from int -> words

def decode_review(text):
    '''
    :param text: a list of ints (movie review data - test or training)
    :return: a string (movie review as seen by user)
    '''
    # note: replace int with char '?' if corresponding val (word) does not exist
    return " ".join([reverse_word_index.get(num, "?") for num in text])

#print(decode_review(test_data[15]))
# for review in train_data:
#     print(review[0]==1)

# Step 3e: data pre-processing (e.g. padding)

# Each movie review has unequal lengths, but input and output layer shape need to be pre-specified
# Solution: padding (ref: L47) in training & testing data
# (keras makes this easier)
# note: restrict each review length to 250 (can also try other values, but here following guide)
# note 2: 'padding' = post param. specifies adding 0 (pad value defined on L49) to END of review

# output of redefined train/test data: 2d numpy array # dimension = (# of reviews * `maxlen` argument)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"],
                                                      padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"],
                                                      padding = "post", maxlen = 250)

# print(len(test_data[0]), len(train_data[15]), len(test_data[5999]))


## Step 4: model train (with validation) & test

# Step 4a: define the model

model = keras.Sequential()
# layer 1: word embedding layer - map each word to a word vector in high-dimensional space (16-dimensional picked arbitrarily here)
#       e.g. group synonyms together (e.g. word vector of 'amazing' and that of 'awesome' are made closer together,
#           compared to that of 'amazing' vs. that of 'shitty')
# layer 2: dimensional-reducing layer (since layer 1 has 16 dimensions)
# layers 3 & 4: dense, fully connected hidden layer & outer layer
model.add(keras.layers.Embedding(10000,16)) # 1st arg.: input_dim - 10000 words in vocabulary on L15
                                            # 2nd arg. : output_dim - 16-dim vector space on which each word is embedded
                                            # layer returns 2D array of dimension 250 by 16, since have 250 words per review,
                                            #, and each rep. as a 16-dimensional vector
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = 'relu'))
model.add(keras.layers.Dense(1,activation = 'sigmoid')) # output layer # sigmoid function gives value between 0 and 1
                                                        # closer to 0 -> model says more likely to be (-) review
                                                       # closer to 1 -> '' (+) review
# model.summary()  # prints a summary of the model

# Step 4b: compile & train model

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# split training data into 2 sets: validation data & training data
# note: test data unchanged
# validation data - a set of data that model haven't seen b4 - used to check how well model is performing based on tuning training data
# note: original training data has 25,000 reviews

x_validation = train_data[:10000] # validate on 10,000 reviews
x_train = train_data[10000:] # train on 15,000 reviews

y_validation = train_labels[:10000]
y_train = train_labels[10000:]

# fit model

# note: An epoch is comprised of one or more batches.
# note2: both epochs & batch sizes are hyperparams. for learning algorithm (e.g. gradient descent), NOT the internal model params.
#   -diff bw epoch vs. batch size: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
fitModel = model.fit(x_train, y_train, epochs = 40, # epochs = # of complete passes of entire training set into NN
                     batch_size = 512, # batch size = # of movie reviews passed into NN
                                                      # before model params. are updated
                     validation_data = (x_validation, y_validation),
                     verbose = 1)

results = model.evaluate(test_data, test_labels)
# print(results) # loss ~ 0.34 # accuracy ~ 0.87

# Step 4d: save model (save time so doesn't have to retrain same model each time script executes)
# note: model loaded in Step 5 for prediction

# note: here only save the model when it is finished training
#   but other methods exist (e.g. checkpointing) that can save model halfway through training

# Calling 'model.save()' creates a h5 file `movie_model.h5` (binary data)
model.save("movie_model.h5")

## Step 5

# Step 5a: predict single review using model

# note: model.predict() takes in 2D numpy array w shape (1,250)
# troubleshooted using: https://stackoverflow.com/questions/53979199/tensorflow-keras-returning-multiple-predictions-while-expecting-one
# predict = model.predict(np.array([test_data[0]]))
# #print(len(predict))
# print()
# print("Review: ")
# print(decode_review(test_data[0])) # output: review in human-readable form
# print()
# print("Predicted by model: " + str(predict))
# print("Actual: " + str(test_labels[0:1]))

# predict 2 test reviews using model
# predict2 = model.predict(np.array([test_data[0], test_data[5]]))
# print("Predicted by model: " + str(predict2))
# print("Actual: " + str(test_labels[0:2]))


## Step 5b: predict using a review not part of the data set!

# -convert review from large string -> encoded list of integers to feed into NN - see function `review_encode'
# -must also make sure length of movie review = 250 words (since that's what model defined above is expecting as input)

def review_encode(list_of_words):
    '''
    :param list_of_words: list of type str # e.g. ['how', 'are', 'you']
    :return: list of type int
    '''
    # note: we don't want to come up with our own dict here, but rather use 'word_index' bult-in for imdb data (Line 34)
    #       and use value 2 to denote words not in the dict (<UNK>)
    list_encoded = [1] # starting value = 1 as defined for model (L52)
    for word in list_of_words:
        # if word already in word_index (make sure to convert to lower-cased version of word since all words in word_index are lower-cased by convention)
        if word.lower() in word_index:
            list_encoded.append(word_index[word.lower()])
        # if not, denote as unknown word (value = 2) as on L53
        else:
            list_encoded.append(2)
    return list_encoded

#############################################################################################
# Note: comment out all lines from Step 1 to Step 4 after executing saving model in L153  ##
##########################################################################################

# load previously saved (Step 4) model
model = keras.models.load_model("movie_model.h5")

# note: convert .rtf -> .txt files if using textEdit app: https://www.techwalla.com/articles/how-to-create-a-txt-file-with-a-mac
# note 2: also try with 'lionking.txt' in directory
with open("1917_rev.txt", encoding = "utf-8") as f:
    for line in f.readlines():
        # split each word in review:
        # - get rid of punctuations with 'replace()'
        # - then get rid of trailing spaces with 'strip()'
        # - finally use 'split()' to convert into a list of strings e.g. ['how', 'are', 'you']
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace("?","").replace("!","").\
                replace(")", "").replace("'", "").replace(";", "").replace(":","").replace("\"","").\
            strip().split()
        # return encoded list (list of ints)
        encode = review_encode(nline)
        #  fix review at length 250 words (like before) - use padding if necessary
        encode = keras.preprocessing.sequence.pad_sequences([encode], value = word_index['<PAD>'],
                                                            padding = 'post', maxlen = 250)
        predict = model.predict(encode) # feed in 2D numpy array # dim = (1,250)
        print(line)
        print(encode)
        print(predict)




