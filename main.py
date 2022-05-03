import time
import os
from turtle import delay

from opt_einsum.backends import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras import layers
from keras.layers import TextVectorization
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
word_vec_options = [50, 100, 200, 300]


def generate_siamese_model(x_train, y_train, x_test, y_test):
    # Define the tensors for the two input images
    left_input = keras.Input(shape=(None,))
    right_input = keras.Input(shape=(None,))
    max_features = 20000  # Only consider the top 20k words
    word_vec_len = word_vec_options[3]

    # ---------- glove embedding ----------
    vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=word_vec_len)
    text_ds = tensorflow.data.Dataset.from_tensor_slices(x_train[:, 1]).batch(128)# TODO see if we can get the words from the other datasets in here
    vectorizer.adapt(text_ds)

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    # TODO this will have to be changed locally
    # get the glove library at http://nlp.stanford.edu/data/glove.6B.zip
    # its a pretty beefy file (abt 2gb fully extracted)
    # path_to_glove_file = os.path.join( os.path.expanduser("~"), "D:/random big files/Machine Learning/glove.6B.100d") #Beef path
    filepath = "glove.6B." + str(word_vec_len) + "d.txt"
    embeddings_index = {}
    f = open(filepath, encoding='utf-8')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        # print("Word:", word, "Coefs:", coef)
        embeddings_index[word] = coef

    print("Found %s word vectors." % len(embeddings_index))

    num_tokens = len(voc) + 2
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, word_vec_len))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    # Input for variable-length sequences of integers
    encoder_model = keras.Sequential()
    # Embed each integer in a glove vector
    encoder_model.add(Embedding(num_tokens, word_vec_len,
                                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                trainable=False, ))
    # Add bidirectional LSTMs w/ decreasing dimentionality
    encoder_model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    encoder_model.add(layers.Bidirectional(layers.LSTM(64)))
    encoder_model.add(layers.Bidirectional(layers.LSTM(32)))

    encoded_l = encoder_model(left_input)
    encoded_r = encoder_model(right_input)

    DNN = layers.Dense(50, activation="sigmoid")([encoded_l, encoded_r])
    DNN = layers.Dense(50, activation="sigmoid")(DNN)
    DNN = layers.Dense(50, activation="sigmoid")(DNN)
    DNN = layers.Dense(10, activation="sigmoid")(DNN)

    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(DNN)

    siam_model = keras.Model(inputs=[left_input, right_input], outputs=outputs)
    siam_model.summary()

    # return the model
    return siam_model


# Handle the input of data from test or train
def loadData(filepath, isTrain):
    # ingest a CSV
    df = pd.read_csv(filepath)
    # process
    wordmap = pd.read_csv("final_proj_CPCmap.csv")
    for index, row in wordmap.iterrows():   #iterate the wordmap, replace any matches
        code = row[0]
        val = row[1]
        df.replace(to_replace=code, value=val, inplace=True)

    targets = np.atleast_2d(np.array(df["target"])).T
    anchors = np.atleast_2d(np.array(df["anchor"])).T
    contexts = np.atleast_2d(np.array(df["context"])).T
    # return a numpy array
    nparr = np.hstack((anchors, targets, contexts))

    # stack a score column if isTrain (test data has no score)
    if isTrain:
        score = np.atleast_2d(np.array(df["score"])).T
        print(score.shape)
        nparr = np.hstack((nparr, score))
    print(nparr)

    # print (nparr)
    print(nparr)

    if isTrain:
        pass
    return nparr


def DNN_main(x_train, y_train, x_test, y_test):
    print(len(x_train), "Training sequences")
    print(len(x_test), "Validation sequences")

    # DNN setup
    maxlen = 10  # Only consider the first 200 words of each movie review

    x_train[0] = keras.preprocessing.sequence.pad_sequences(x_train[0], maxlen=maxlen)
    x_train[1] = keras.preprocessing.sequence.pad_sequences(x_train[1], maxlen=maxlen)
    x_test[0] = keras.preprocessing.sequence.pad_sequences(x_test[0], maxlen=maxlen)
    x_test[1] = keras.preprocessing.sequence.pad_sequences(x_test[1], maxlen=maxlen)

    model = generate_siamese_model(x_train, y_train, x_test, y_test)

    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit([x_train[0], x_train[1]], y_train, batch_size=32, epochs=2, validation_data=([x_train[0], x_train[1]], y_test))


if __name__ == "__main__":
    # Load data
    # images = np.load("fashion_mnist_{}_images.npy".format(which)) / 255.
    # labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    train = loadData("train.csv", True)  # make sure this is inside the repo on your local - it's in the gitignore
    test = loadData("test.csv", False)  # make sure this is inside the repo on your local - it's in the gitignore
    train_X = train[:, :-1]
    train_Y = train[:, -1]
    print("Y shape is: ", train_Y.shape)
    # ^ Before Validation
    n = train_X.shape[0]
    # print("Sample size is: ", n)
    validation_n = int(n * 0.2)
    validation_indices = np.arange(validation_n)
    np.random.shuffle(validation_indices)
    # validation_indices = np.random.shuffle(validation_n)
    train_X_rm = np.delete(train_X, validation_indices)#TODO fix this, output is 1D
    train_Y_rm = np.delete(train_Y, validation_indices)

    test_X = train_X[validation_indices]  # there is no Y data (no score) for test data in this set
    test_Y = train_Y[validation_indices]  #
    # Shallow Model

    # Shallow keras

    # Main Keras (Deep)
    DNN_main(train_X_rm, train_Y_rm  , test_X, test_Y)
    # Output what we need for the submission
