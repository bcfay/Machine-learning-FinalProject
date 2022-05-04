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
maxlen = 5  # pad length


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tensorflow.math.reduce_sum(tensorflow.math.square(x - y), axis=1, keepdims=True)
    return tensorflow.math.sqrt(tensorflow.math.maximum(sum_square, tensorflow.keras.backend.epsilon()))


def generate_siamese_model(x_train, x_test):
    # Define the tensors for the two input images
    left_input = keras.Input(maxlen, dtype='int32')
    right_input = keras.Input(maxlen, dtype='int32')
    context_input = keras.Input(maxlen, dtype='int32')
    max_features = 400000  # Only consider the top 20k words
    word_vec_len = word_vec_options[3]

    # ---------- glove embedding ----------
    vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=word_vec_len)
    hbug = np.hstack((x_train[0:3], x_test[0:3])).flatten().tolist()
    lean_bug = []
    for i, word in enumerate(hbug):
        if (word == ''):
            pass
        else:
            lean_bug.append(word)

    # text_ds = tensorflow.data.Dataset.from_tensor_slices(hbug).batch(
    #     128)  # TODO see if we can get the words from the other datasets in here
    print("Vectorizor adaption.")
    vectorizer.adapt(lean_bug)

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    # TODO this will have to be changed locally
    # get the glove library at http://nlp.stanford.edu/data/glove.6B.zip
    # its a pretty beefy file (abt 2gb fully extracted)
    # path_to_glove_file = os.path.join( os.path.expanduser("~"), "D:/random big files/Machine Learning/glove.6B.100d") #Beef path
    print("Glove reading.")
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

    encoder_model = keras.Sequential()
    # Embed each integer in a glove vector
    encoder_model.add(Embedding(num_tokens, word_vec_len,
                                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                trainable=False))
    # Add bidirectional LSTMs w/ decreasing dimentionality
    encoder_model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
    encoder_model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    encoder_model.add(layers.Bidirectional(layers.LSTM(128)))

    context_encoder_model = keras.Sequential()
    context_encoder_model.add(Embedding(num_tokens, word_vec_len,
                                        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                        trainable=False))
    context_encoder_model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
    context_encoder_model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    context_encoder_model.add(layers.Bidirectional(layers.LSTM(128)))

    encoder_model.summary()

    # [print(i.shape, i.dtype) for i in encoder_model.inputs]
    # [print(o.shape, o.dtype) for o in encoder_model.outputs]
    # [print(l.name, l.input_shape, l.dtype) for l in encoder_model.layers]

    context_encoder_model.summary()
    #
    # [print(i.shape, i.dtype) for i in context_encoder_model.inputs]
    # [print(o.shape, o.dtype) for o in context_encoder_model.outputs]
    # [print(l.name, l.input_shape, l.dtype) for l in context_encoder_model.layers]

    encoded_l = encoder_model(left_input)
    encoded_r = encoder_model(right_input)
    encoded_c = context_encoder_model(context_input)

    merge_layer = layers.Lambda(euclidean_distance)([encoded_l, encoded_r])

    merged = keras.layers.Concatenate(axis=1)([merge_layer, encoded_c])
    dropout_rate = .01
    DNN = layers.Dense(200, activation="tanh")(merged)
    DNN = layers.Dropout(dropout_rate, input_shape=(100,))(DNN)
    DNN = layers.Dense(200, activation="tanh")(DNN)
    DNN = layers.Dropout(dropout_rate, input_shape=(100,))(DNN)
    DNN = layers.Dense(50, activation="tanh")(DNN)
    DNN = layers.Dropout(dropout_rate, input_shape=(50,))(DNN)
    DNN = layers.Dense(30, activation="tanh")(DNN)
    DNN = layers.Dropout(dropout_rate, input_shape=(30,))(DNN)
    DNN = layers.Dense(30, activation="tanh")(DNN)
    DNN = layers.Dropout(dropout_rate, input_shape=(30,))(DNN)

    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(DNN)
    # outputs = tensorflow.round(outputs)

    siam_model = keras.Model(inputs=[left_input, right_input, context_input], outputs=outputs)
    siam_model.summary()

    # [print(i.shape, i.dtype) for i in siam_model.inputs]
    # [print(o.shape, o.dtype) for o in siam_model.outputs]
    # [print(l.name, l.input_shape, l.dtype) for l in siam_model.layers]

    # return the model
    return siam_model, voc


def generate_siamese_model_shallow(x_train, x_test):
    # Define the tensors for the two input images
    left_input = keras.Input(maxlen, dtype='int32')
    right_input = keras.Input(maxlen, dtype='int32')
    context_input = keras.Input(maxlen, dtype='int32')
    max_features = 400000  # Only consider the top 20k words
    word_vec_len = word_vec_options[3]

    # ---------- glove embedding ----------
    vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=word_vec_len)
    hbug = np.hstack((x_train[0:3], x_test[0:3])).flatten().tolist()
    lean_bug = []
    for i, word in enumerate(hbug):
        if (word == ''):
            pass
        else:
            lean_bug.append(word)

    # text_ds = tensorflow.data.Dataset.from_tensor_slices(hbug).batch(
    #     128)  # TODO see if we can get the words from the other datasets in here
    print("Vectorizor adaption.")
    vectorizer.adapt(lean_bug)

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    # TODO this will have to be changed locally
    # get the glove library at http://nlp.stanford.edu/data/glove.6B.zip
    # its a pretty beefy file (abt 2gb fully extracted)
    # path_to_glove_file = os.path.join( os.path.expanduser("~"), "D:/random big files/Machine Learning/glove.6B.100d") #Beef path
    print("Glove reading.")
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

    encoder_model = keras.Sequential()
    # Embed each integer in a glove vector
    encoder_model.add(Embedding(num_tokens, word_vec_len,
                                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                trainable=False))
    # Add bidirectional LSTMs w/ decreasing dimentionality
    encoder_model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
    encoder_model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    encoder_model.add(layers.Bidirectional(layers.LSTM(128)))

    context_encoder_model = keras.Sequential()
    context_encoder_model.add(Embedding(num_tokens, word_vec_len,
                                        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                        trainable=False))
    context_encoder_model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
    context_encoder_model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    context_encoder_model.add(layers.Bidirectional(layers.LSTM(128)))

    encoder_model.summary()

    # [print(i.shape, i.dtype) for i in encoder_model.inputs]
    # [print(o.shape, o.dtype) for o in encoder_model.outputs]
    # [print(l.name, l.input_shape, l.dtype) for l in encoder_model.layers]

    context_encoder_model.summary()
    #
    # [print(i.shape, i.dtype) for i in context_encoder_model.inputs]
    # [print(o.shape, o.dtype) for o in context_encoder_model.outputs]
    # [print(l.name, l.input_shape, l.dtype) for l in context_encoder_model.layers]

    encoded_l = encoder_model(left_input)
    encoded_r = encoder_model(right_input)
    encoded_c = context_encoder_model(context_input)

    merged = keras.layers.Concatenate(axis=1)([left_input, right_input, context_input])
    dropout_rate = .01
    # DNN = layers.Dense(200, activation="tanh")(merged)
    # DNN = layers.Dropout(dropout_rate,input_shape=(100,))(DNN)
    # DNN = layers.Dense(200, activation="tanh")(DNN)
    # DNN = layers.Dropout(dropout_rate,input_shape=(100,))(DNN)
    # DNN = layers.Dense(50, activation="tanh")(DNN)
    # DNN = layers.Dropout(dropout_rate,input_shape=(50,))(DNN)
    # DNN = layers.Dense(30, activation="tanh")(DNN)
    # DNN = layers.Dropout(dropout_rate,input_shape=(30,))(DNN)
    # DNN = layers.Dense(30, activation="tanh")(DNN)
    # DNN = layers.Dropout(dropout_rate,input_shape=(30,))(DNN)

    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(merged)
    # outputs = tensorflow.round(outputs)

    siam_model = keras.Model(inputs=[left_input, right_input, context_input], outputs=outputs)
    siam_model.summary()

    # [print(i.shape, i.dtype) for i in siam_model.inputs]
    # [print(o.shape, o.dtype) for o in siam_model.outputs]
    # [print(l.name, l.input_shape, l.dtype) for l in siam_model.layers]

    # return the model
    return siam_model, voc


# TODO make words separated by dashes a single word. They count as one in GLOVE but end up as 3 differnet words in our data
# Handle the input of data from test or train
def loadData(filepath, isTrain):
    # ingest a CSV
    df = pd.read_csv(filepath)
    # process
    wordmap = pd.read_csv("final_proj_CPCmap.csv")
    for index, row in wordmap.iterrows():  # iterate the wordmap, replace any matches
        code = row[0]
        val = row[1]
        if code == "B01" or code == "B01,":
            manualList = ['PHYSICAL', 'OR', 'CHEMICAL', 'PROCESSES', 'OR', 'APPARATUS', 'IN', 'GENERAL']
            df.replace(to_replace=code, value=manualList, inplace=True)
            pass
        df.replace(to_replace=code, value=val, inplace=True)
    rawContexts = df["context"]
    listOfContextStrings = []
    # TODO make this a helper function
    for index, expression in enumerate(rawContexts):
        expression = expression.replace("[", '')
        expression = expression.replace("]", '')
        expression = expression.replace('\'', '')
        expression = expression.replace('\"', '')
        expression = expression.replace(',', '')
        expression = expression.replace('-', ' ')
        expression = expression.replace(' - ', ' ')
        expression = expression.lower()
        words = np.array(expression.split(' '))
        # print(words)

        listOfContextStrings.append(words)
        # allWords.extend(words)

    raw_targets = df["target"]
    listOfTargetStrings = []
    for index, expression in enumerate(raw_targets):
        expression = expression.replace("[", '')
        expression = expression.replace("]", '')
        expression = expression.replace('\'', '')
        expression = expression.replace('\"', '')
        expression = expression.replace(',', '')
        expression = expression.replace('-', ' ')
        expression = expression.replace(' - ', ' ')
        expression = expression.lower()
        words = np.array(expression.split(' '))
        # print(words)
        listOfTargetStrings.append(words)

    raw_anchors = df["anchor"]
    listOfAnchorStrings = []
    for index, expression in enumerate(raw_anchors):
        expression = expression.replace("[", '')
        expression = expression.replace("]", '')
        expression = expression.replace('\'', '')
        expression = expression.replace('\"', '')
        expression = expression.replace(' - ', ' ')
        expression = expression.replace('-', ' ')
        expression = expression.replace(',', '')
        expression = expression.lower()
        words = np.array(expression.split(' '))
        # print(words)
        listOfAnchorStrings.append(words)

    targets = keras.preprocessing.sequence.pad_sequences(np.array(listOfTargetStrings, dtype=object), maxlen=maxlen,
                                                         dtype=object, value='', truncating='post')
    anchors = keras.preprocessing.sequence.pad_sequences(np.array(listOfAnchorStrings, dtype=object), maxlen=maxlen,
                                                         dtype=object, value='', truncating='post')
    contexts = keras.preprocessing.sequence.pad_sequences(np.array(listOfContextStrings, dtype=object), maxlen=maxlen,
                                                          dtype=object, value='', truncating='post')

    # TODO make data have phrases as lits of strings, not a single string. This will add another dimention to the data.
    # data_len = len(targets)
    #
    # split_targets =  np.atleast_2d(np.empty(data_len))
    # split_anchors =  np.atleast_2d(np.empty(data_len))
    # split_contexts =  np.atleast_2d(np.empty(data_len))
    #
    # for i in range(data_len):
    #     split_targets[i] = np.char.split(targets[i][0])
    #     split_anchors[i] = np.char.split(anchors[i][0])
    #     split_contexts[i] = np.char.split(contexts[i][0])
    # return a numpy array

    # stack a score column if isTrain (test data has no score)
    if isTrain:
        score = np.atleast_2d(np.array(df["score"])).T
        print(score.shape)
        nparr = (np.vstack(([anchors], [targets], [contexts])), score)
    else:
        nparr = np.vstack(([anchors], [targets], [contexts]))
    print(nparr)
    ids = df["id"]
    return nparr, ids


def NN_3layer(x_train, y_train, x_test, y_test):
    # DNN setup
    print("x_train:\n", x_train)
    print("y_train:\n", y_train)
    print("x_test:\n", x_test)
    print("x_train:\n", x_train, "y_train", y_train, "x_test", x_test, "y_test", y_test)
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

    model, voc = generate_siamese_model_shallow(x_train, x_test)
    temp_x_train = np.empty_like(x_train, dtype=int)
    temp_x_test = np.empty_like(x_test, dtype=int)

    # TODO vectorize for speed
    for i, col in enumerate(x_train):
        for j, sample in enumerate(col):
            for k, word in enumerate(sample):
                try:
                    temp_x_train[i, j, k] = voc.index(word)
                except:
                    temp_x_train[i, j, k] = 1

    for i, col in enumerate(x_test):
        for j, sample in enumerate(col):
            for k, word in enumerate(sample):
                try:
                    temp_x_test[i, j, k] = voc.index(word)
                except:
                    temp_x_test[i, j, k] = 1

    temp_x_train_t = tensorflow.convert_to_tensor(temp_x_train[0].tolist(), dtype='int32')
    temp_x_train_a = tensorflow.convert_to_tensor(temp_x_train[1].tolist(), dtype='int32')
    temp_x_train_c = tensorflow.convert_to_tensor(temp_x_train[2].tolist(), dtype='int32')
    temp_y_train = tensorflow.convert_to_tensor(y_train[:, 0].tolist(), dtype="float32")
    temp_x_test_a = tensorflow.convert_to_tensor(temp_x_test[0].tolist(), dtype='int32')
    temp_x_test_t = tensorflow.convert_to_tensor(temp_x_test[1].tolist(), dtype='int32')
    temp_x_test_c = tensorflow.convert_to_tensor(temp_x_test[2].tolist(), dtype='int32')
    temp_y_test = tensorflow.convert_to_tensor(y_test[:, 0].tolist(), dtype='float32')
    print("Compiling.")
    model.compile("adam", "mean_squared_error", metrics=["accuracy", "binary_accuracy"])
    print("Fiting.")
    model.fit([temp_x_train_t, temp_x_train_a, temp_x_train_c], temp_y_train, batch_size=2000, epochs=2,
              validation_data=([temp_x_test_a, temp_x_test_t, temp_x_test_c], temp_y_test), verbose=1)
    print("Predicting.")
    pred = model.predict([temp_x_test_a, temp_x_test_t, temp_x_test_c])
    worst_num = 5
    worst = [[pred[:worst_num, 0] - y_test[:worst_num, 0]], [x_test[:worst_num, i]]]
    for i in range(y_test.shape[1]):
        delta = pred[i, 0] - y_test[i, 0]
        print("prediction:", pred[i][0], "truth:", y_test[i][0], "delta:", delta)
        print("anchor data:", x_test[0, i], "target:", x_test[1, i], "context:", x_test[2, i])
        for i, worst_delta in enumerate(worst[0]):
            if delta > worst_delta:
                worst[0, i] = delta
                worst[1, i] = x_test[:, i]

    print(worst)
    model.save('my_model')
    return model


def DNN_main(x_train, y_train, x_test, y_test):
    # DNN setup
    print("x_train:\n", x_train)
    print("y_train:\n", y_train)
    print("x_test:\n", x_test)
    print("x_train:\n", x_train, "y_train", y_train, "x_test", x_test, "y_test", y_test)
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

    model, voc = generate_siamese_model(x_train, x_test)
    temp_x_train = np.empty_like(x_train, dtype=int)
    temp_x_test = np.empty_like(x_test, dtype=int)

    # TODO vectorize for speed
    for i, col in enumerate(x_train):
        for j, sample in enumerate(col):
            for k, word in enumerate(sample):
                try:
                    temp_x_train[i, j, k] = voc.index(word)
                except:
                    temp_x_train[i, j, k] = 1

    for i, col in enumerate(x_test):
        for j, sample in enumerate(col):
            for k, word in enumerate(sample):
                try:
                    temp_x_test[i, j, k] = voc.index(word)
                except:
                    temp_x_test[i, j, k] = 1

    temp_x_train_t = tensorflow.convert_to_tensor(temp_x_train[0].tolist(), dtype='int32')
    temp_x_train_a = tensorflow.convert_to_tensor(temp_x_train[1].tolist(), dtype='int32')
    temp_x_train_c = tensorflow.convert_to_tensor(temp_x_train[2].tolist(), dtype='int32')
    temp_y_train = tensorflow.convert_to_tensor(y_train[:, 0].tolist(), dtype="float32")
    temp_x_test_a = tensorflow.convert_to_tensor(temp_x_test[0].tolist(), dtype='int32')
    temp_x_test_t = tensorflow.convert_to_tensor(temp_x_test[1].tolist(), dtype='int32')
    temp_x_test_c = tensorflow.convert_to_tensor(temp_x_test[2].tolist(), dtype='int32')
    temp_y_test = tensorflow.convert_to_tensor(y_test[:, 0].tolist(), dtype='float32')
    print("Compiling.")
    model.compile("adam", "mean_squared_error", metrics=["accuracy", "binary_crossentropy", "mean_squared_error"])
    print("Fiting.")
    model.fit([temp_x_train_t, temp_x_train_a, temp_x_train_c], temp_y_train, batch_size=2000, epochs=500,
              validation_data=([temp_x_test_a, temp_x_test_t, temp_x_test_c], temp_y_test), verbose=1)
    print("Predicting.")
    pred = model.predict([temp_x_test_a, temp_x_test_t, temp_x_test_c])
    worst_num = 10
    worst_preds = np.zeros(worst_num)
    worst_truths = np.zeros(worst_num)
    worst_delta = np.zeros(worst_num)
    worst_data = x_test[:, :worst_num]
    for i in range(y_test.shape[0]):
        delta = pred[i, 0] - y_test[i, 0]
        entered = False
        for j in range(worst_num):
            if (delta > worst_delta[j]) and not entered:
                worst_delta[j] = delta
                worst_preds[j] = pred[i, 0]
                worst_truths[j] = y_test[i, 0]
                worst_data[:, j] = x_test[:, i]
                entered = True

    print("Overall worst preds:", worst_preds, "Overall worst truths:", worst_truths, "Overall worst deltas:",
          worst_delta)
    print("anchor data:", worst_data[0, :])
    print("target data:", worst_data[1, :])
    print("context data:", worst_data[2, :])

    first_num = 10
    first_preds = np.zeros(first_num)
    first_truths = np.zeros(first_num)
    first_delta = np.zeros(first_num)
    first_data = x_test[:, :first_num]

    for i in range(first_num):
        first_delta[i] = delta
        first_preds[i] = pred[i, 0]
        first_truths[i] = y_test[i, 0]
        first_data[:, i] = x_test[:, i]
        entered = True

    print("First validation data preds:", first_preds, "First truths:", worst_truths, "First deltas:",
          worst_delta)
    print("First anchor data:", first_data[0, :])
    print("First target data:", first_data[1, :])
    print("First context data:", first_data[2, :])

    # model.save('my_model')
    return model


if __name__ == "__main__":
    # Load data
    # images = np.load("fashion_mnist_{}_images.npy".format(which)) / 255.
    # labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    train, trainIDs = loadData("train.csv",
                               True)  # make sure this is inside the repo on your local - it's in the gitignore
    test, testIDs = loadData("test.csv",
                             False)  # make sure this is inside the repo on your local - it's in the gitignore
    train_X = train[0]
    train_Y = train[1]
    print("Y shape is: ", train_Y.shape)
    # ^ Before Validation
    n = train_X.shape[1]
    # print("Sample size is: ", n)
    validation_n = int(n * 0.2)
    potential_indices = np.arange(n)
    validation_indices = []
    train_X_rm = np.empty((3, (n - validation_n), maxlen), dtype=object)
    train_Y_rm = np.empty(((n - validation_n), 1), dtype=object)
    valid_X = np.empty((3, validation_n, maxlen), dtype=object)
    valid_Y = np.empty((validation_n, 1), dtype=object)
    np.random.shuffle(potential_indices)
    temp = np.array(train_X)

    valid_i = 0
    train_i = 0

    for index, potential in enumerate(potential_indices):
        # print(potential)
        potential_val = temp[:, potential, :]
        if index % 5 == 4:
            # delete everything but every 5th - keep the 5th
            valid_X[:, valid_i] = (potential_val)
            valid_Y[valid_i] = (train_Y[potential])
            valid_i += 1
        else:
            train_X_rm[:, train_i] = (potential_val)
            train_Y_rm[train_i] = (train_Y[potential])
            train_i += 1

    # validation_indices = np.random.shuffle(validation_n)
    # train_X_rm = np.delete(train_X, potential_indices, axis=1)  # TODO fix this, output is 1D
    # train_Y_rm = np.delete(train_Y, potential_indices, axis=0)

    # valid_X = train_X[:][potential_indices]  # there is no Y data (no score) for test data in this set
    # valid_Y = train_Y[potential_indices]  #
    # Shallow Model

    # Shallow keras

    # Main Keras (Deep)
    # model = NN_3layer(train_X_rm, train_Y_rm, valid_X, valid_Y)
    model = DNN_main(train_X_rm, train_Y_rm, valid_X, valid_Y)
    # Output what we need for the submission
    # pseudo
    #
    x = test
    test_anchor = tensorflow.convert_to_tensor(x[0].tolist(), dtype='int32')
    test_target = tensorflow.convert_to_tensor(x[1].tolist(), dtype='int32')
    test_context = tensorflow.convert_to_tensor(x[2].tolist(), dtype='int32')

    y = model.predict(([test_anchor, test_target, test_context]))
    df = pd.DataFrame({'id': x, 'score': y})
    df.to_csv("submit_to_kaggle.csv", mode='a', header=False, index=False)
    # df.tocsv()
    # print(model.predict(test))
