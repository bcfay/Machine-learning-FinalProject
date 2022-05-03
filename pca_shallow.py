import numpy as np
import pandas as pd
import math
import sklearn.metrics.pairwise
import sklearn.metrics
from Toolbox import Toolbox
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

toolbox = Toolbox()

def full_process(train_input):
    train = pd.read_csv(train_input)
    train = train.sample(frac=1, random_state=1)
    train['anchor'] = train['anchor'].apply(toolbox.preprocess)
    train['target'] = train['target'].apply(toolbox.preprocess)
    out_df = toolbox.process_df(train, ['anchor', 'target'], ['context'])
    return out_df

def extract_for_pca(input_df):
    df = input_df.copy()
    anchor = []
    target = []
    context = []
    i = 0
    for tuple in df.itertuples():
        if i ==0:
            anchor = tuple[2]
            target = tuple[3]
            context = tuple[4]
            i += 1
        else:
            anchor = np.vstack([anchor, tuple[2]])
            target = np.vstack([target, tuple[3]])
            context = np.vstack([context, tuple[4]])
            i += 1
    return np.hstack([anchor, target, context]), input_df['score']


def grid_search(X, y):
    clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
            ])
    param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ['rbf', 'sigmoid']
    }]
    gsearch = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1)
    gsearch.fit(X, y)


def crude_round(num):
    if num <= 0.125:
        return 0
    if num <= 0.25:
        return 0.25
    if num <= 0.375:
        return 0.25
    if num <= 0.5:
        return 0.5
    if num <= 0.625:
        return 0.5
    if num <= 0.75:
        return 0.75
    if num <= 0.875:
        return 0.75
    else:
        return 1


def guess_run(input):
    square_error = []
    score_list = []
    guess_list = []
    for x in input.itertuples():
        score_num = input.columns.get_loc('score') + 1
        anchor_num = input.columns.get_loc('anchor') + 1
        target_num = input.columns.get_loc('target') + 1
        score = x[score_num]
        score_list.append(score)
        anchor_vector = toolbox.tfidf.transform([x[anchor_num]])
        target_vector = toolbox.tfidf.transform([x[target_num]])
        guess = sklearn.metrics.pairwise.cosine_similarity(anchor_vector, target_vector).tolist()[0][0]
        guess_list.append(guess)
        square_error.append(math.pow((guess - score), 2))
    MSE = (np.mean(square_error))
    return MSE, guess_list, score_list


def get_guess(train, test):
    test['anchor'] = test['anchor'].apply(toolbox.preprocess)
    test['target'] = test['target'].apply(toolbox.preprocess)

    out1 = (train['anchor'].tolist())
    out2 = (train['target'].tolist())
    out3 = (test['anchor'].tolist())
    out4 = (test['target'].tolist())
    out = out1 + out2 + out3 + out4
    toolbox.tfidf.fit(out)
    guess_list = guess_run(test)
    id = test['id'].to_numpy()[:, np.newaxis]

    guess = np.array(guess_list)
    guess_df = pd.DataFrame(guess, columns=['guess'])
    guess_df['guess'] = guess_df['guess'].apply(crude_round)
    guess_np = guess_df['guess'].to_numpy()[:, np.newaxis]

    out_np = np.hstack([id, guess_np])
    out_df = pd.DataFrame(out_np, columns=['id', 'score'])
    return out_df


def run_pca(train_input):
    train = pd.read_csv(train_input)
    train = train.sample(frac=1, random_state=1).reset_index()

    phrase_encoder = OrdinalEncoder()
    context_encoder = OrdinalEncoder()
    y_encoder = OrdinalEncoder()

    phrase_encoder.fit(np.array(((train['anchor'].tolist()) + (train['target'].tolist()))).reshape(-1, 1))
    context_encoder.fit(train['context'].to_numpy().reshape(-1, 1))
    y_encoder.fit(train['score'].to_numpy().reshape(-1, 1))

    encoded_anchor = phrase_encoder.transform(train['anchor'].to_numpy().reshape(-1, 1))
    encoded_target = phrase_encoder.transform(train['target'].to_numpy().reshape(-1, 1))
    encoded_context = context_encoder.transform(train['context'].to_numpy().reshape(-1, 1))
    encoded_score = y_encoder.transform(train['score'].to_numpy().reshape(-1, 1))

    for tuple in train.itertuples():
        index = tuple[0]
        train.at[index, 'anchor'] = encoded_anchor[index][0]

    for tuple in train.itertuples():
        index = tuple[0]
        train.at[index, 'target'] = encoded_target[index][0]

    for tuple in train.itertuples():
        index = tuple[0]
        train.at[index, 'context'] = encoded_context[index][0]

    for tuple in train.itertuples():
        index = tuple[0]
        train.at[index, 'score'] = encoded_score[index][0]

    y = train['score']
    train = train.drop(columns=['index', 'id', 'score'])

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(train.copy())
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
    plt.show()

    kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=10, alpha=0.1)
    X_reduced = kernel_pca.fit_transform(train.copy())
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
    plt.show()

    clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

    param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

    grid_search = GridSearchCV(clf, param_grid, cv=3)

    grid_search.fit(train.copy(), y.copy())


