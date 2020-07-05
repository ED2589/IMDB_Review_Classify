'''
Sentiment analysis project: classifying movie review as (+) or (-)
Following 4-part guide: https://techwithtim.net/tutorials/python-neural-networks/text-classification-p1/
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
# Note: model trained with vocab. of 88000 words (instead of 10000 words like in 'textClassify_imdb.py') to improve
#       accuracy of prediction
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 88000)
word_index = data.get_word_index()
word_index =  {key: (val+3) for key, val in word_index.items()}
word_index["<PAD>"] = 0 # default start char # see doc 'start_char': https://keras.io/api/datasets/imdb/#get_word_index-function
word_index["<START>"] = 1 # all training/testing data starts (type list) starts with 1, so we encode as 'START' of review
word_index["<UNK>"] = 2 # took words out when loading data (L15) & default replacement in data is integer '2'  see doc 'oov_char': https://keras.io/api/datasets/imdb/#get_word_index-function
word_index["<UNUSED>"] = 3
reverse_word_index = {val:key for key, val in word_index.items()}

def decode_review(text):
    '''
    :param text: a list of ints (movie review data - test or training)
    :return: a string (movie review as seen by user)
    '''
    # note: replace int with char '?' if corresponding val (word) does not exist
    return " ".join([reverse_word_index.get(num, "?") for num in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"],
                                                      padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"],
                                                      padding = "post", maxlen = 250)
#
# model = keras.Sequential()
# model.add(keras.layers.Embedding(88000,16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation = 'relu'))
# model.add(keras.layers.Dense(1,activation = 'sigmoid')) # output layer # sigmoid function gives value between 0 and 1
#
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
# # print(model.summary())
# #
# # print(decode_review(test_data[15]))
# # print(reverse_word_index[9459])
#
# x_validation = train_data[:10000] # validate on 10,000 reviews
# x_train = train_data[10000:] # train on 15,000 reviews
#
# y_validation = train_labels[:10000]
# y_train = train_labels[10000:]
# #
# fitModel = model.fit(x_train, y_train, epochs = 40, # epochs = # of complete passes of entire training set into NN
#                      batch_size = 512, # batch size = # of movie reviews passed into NN
#                                                       # before model params. are updated
#                      validation_data = (x_validation, y_validation),
#                      verbose = 1)
#
# results = model.evaluate(test_data, test_labels)

################
## save model ##
###############
# model.save("movie_model.h5")

################
## load model ##
###############
model = keras.models.load_model("movie_model.h5")


# #
#
# predict = model.predict(np.array([test_data[0]]))
# print()
# print("Review: ")
# print(decode_review(test_data[0])) # output: review in human-readable form
# print()
# print("Predicted by model: " + str(predict))
# print("Actual: " + str(test_labels[0:1]))
#
# #redict 2 test reviews using model
# predict2 = model.predict(np.array([test_data[0], test_data[5]]))
# print("Predicted by model: " + str(predict2))
# print("Actual: " + str(test_labels[0:2]))


# function that converts list of strings -> encoded list of integers
# i.e. human readable review  -> encoded list of int that neural net takes in as input layer
def review_encode(list_of_words):
    '''
    :param list_of_words: list of type str # e.g. ['how', 'are', 'you']
    :return: list of type int
    '''
    # note: we don't want to come up with our own dict here, but rather use 'word_index' bult-in for imdb data (L34)
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

# note: also try with 'lionking.txt'
with open("1917_rev.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


