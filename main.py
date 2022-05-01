import pandas as pd
import numpy as np
import time


# Handle the input of data from test or train
def loadData(filepath, isTrain):
    # ingest a CSV
    df = pd.read_csv(filepath)

    # process
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


if __name__ == "__main__":
    # Load data
    # images = np.load("fashion_mnist_{}_images.npy".format(which)) / 255.
    # labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    train = loadData("train.csv", True)  # make sure this is inside the repo on your local - it's in the gitignore
    test = loadData("test.csv", False)  # make sure this is inside the repo on your local - it's in the gitignore
    train_X = train[:, :-1]
    train_Y = train[:, -1]
    testX = test  # there is no Y data (no score) for test data in this set

    # Shallow Model

    # Shallow keras

    # Main Keras (Deep)

    # Output what we need for the submission
