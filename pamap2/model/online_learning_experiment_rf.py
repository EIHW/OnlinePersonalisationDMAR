import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
import numpy as np
from load_data import leave_user_out_data
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow as tf

#data_path_split = '/mnt/data/hand-activity-data/svm_features/user1_model_split/'
#data_path_y_split = '/mnt/data/hand-activity-data/split/'

use_labels = True

activities = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    ## 9,
    ## 10,
    ## 11,
    12,
    13,
    16,
    17,
    ## 18,
    ## 19,
    ## 20,
    24,
    ## 0,
]


def load_train_data():
    x_train = np.load('data/pamap2/train.npy')
    # _, y_train, _, _ = leave_user_out_data(5)
    y_train = np.load('data/pamap2/train_y.npy')
    return x_train, y_train


def split_test_set(x, y):
    x_online = []
    y_online = []
    x_test = []
    y_test = []
    for i in range(0, 12):
        indixes = [label == i for label in y]
        x_activity = x[indixes]
        y_activity = y[indixes]

        middle = int(len(x_activity)/2)
        x_online.append(x_activity[:middle])
        x_test.append(x_activity[middle:])
        y_online.append(y_activity[:middle])
        y_test.append(y_activity[middle:])

    x_online = np.concatenate(x_online)
    y_online = np.concatenate(y_online)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    return x_online, y_online, x_test, y_test


def load_test_data():
    x_online = np.load('data/pamap2/test.npy')
    # _, _, _, y_online = leave_user_out_data(5)
    y_online = np.load('data/pamap2/test_y.npy')
    #TODO: change back
    x_test, y_test, x_online, y_online = split_test_set(x_online, y_online)

    # online_learning_samples = 4 * 12
    # print(np.unique(y_online, return_counts=True))
    online_learning_samples = 0.1
    _, x_online, _, y_online = train_test_split(x_online, y_online, test_size=online_learning_samples, stratify=y_online)

    print(y_online)

    return x_online, y_online, x_test, y_test


x_train, y_train_value = load_train_data()
x_online, y_online, x_test, y_test = load_test_data()

print('x_train: {}'.format(x_train.shape))
print('x_online: {}'.format(x_online.shape))
print('x_test: {}'.format(x_test.shape))

print('##################')
print('RF')
classifier = RandomForestClassifier(n_estimators=100, warm_start=True, n_jobs=-1)
classifier.fit(x_train, y_train_value)

# use predictions of model instead of actual labels
if not use_labels:
    y_online = classifier.predict(x_online)

accuracy_before_training = classifier.score(x_test, y_test)
print('Accuracy on remaining test set: {}'.format(accuracy_before_training))

classifier.n_estimators = 400
print(f'estimators after: {classifier.n_estimators}')
classifier.fit(x_online, y_online)
accuracy = classifier.score(x_test, y_test)
print('Accuracy on remaining test set: {}'.format(accuracy))

# for n_estimators in range(110, 2000, 50):
#     classifier_tmp = deepcopy(classifier)
#     classifier_tmp.n_estimators = n_estimators
#     print(f'estimators after: {classifier_tmp.n_estimators}')
#     classifier_tmp.fit(x_online, y_online)
#     pred = classifier_tmp.predict(x_test)
#
#     categorical_accuracy = CategoricalAccuracy()
#     categorical_accuracy.update_state(tf.one_hot(y_test, 12), tf.one_hot(pred, 12))
#     accuracy = categorical_accuracy.result().numpy()
#     # accuracy = classifier_tmp.score(x_test, y_test)
#     print('Accuracy on remaining test set: {}'.format(accuracy))
