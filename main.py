import time
import os
import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras import layers
from keras.layers import TextVectorization
from keras.layers import Embedding



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
    # (x_train, y_train), (x_val, y_val) = 0, 0, 0, 0  #
    print(len(x_train), "Training sequences")
    print(len(x_test), "Validation sequences")

    # DNN setup
    max_features = 10  # Only consider the top 20k words
    maxlen = 1  # Only consider the first 200 words of each movie review

    vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=300)
    text_ds = tensorflow.data.Dataset.from_tensor_slices(x_train).batch(128)
    vectorizer.adapt(text_ds)

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))


    # TODO this will have to be changed locally
    # get the glove library at http://nlp.stanford.edu/data/glove.6B.zip
    # its a pretty beefy file (abt 2gb fully extracted)
    # path_to_glove_file = os.path.join( os.path.expanduser("~"), "D:/random big files/Machine Learning/glove.6B.100d") #Beef path
    path_to_glove_file = os.path.join( os.path.expanduser("~"), "D:/random big files/Machine Learning/glove.6B.300d")

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    num_tokens = len(voc) + 2
    embedding_dim = 100
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
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
    inputs = keras.Input(shape=(None,), dtype="int32")

    # Embed each integer in a 128-dimensional vector
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(embedding_layer)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)


    model = keras.Model(inputs, outputs)
    model.summary()



    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))


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
    #print("Sample size is: ", n)
    validation_n = int(n*0.2)
    validation_indices = np.arange(validation_n)
    np.random.shuffle(validation_indices)
    #validation_indices = np.random.shuffle(validation_n)
    train_X = np.delete(train_X, validation_indices)
    train_Y = np.delete(train_Y, validation_indices)

    test_X = train_X[validation_indices]    # there is no Y data (no score) for test data in this set
    test_Y = train_Y[validation_indices]    #
    # Shallow Model

    # Shallow keras

    # Main Keras (Deep)
    DNN_main(train_X, train_Y, test_X, test_Y)
    # Output what we need for the submission

