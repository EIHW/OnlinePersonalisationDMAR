import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import datetime
import numpy as np
import itertools
import random
import json
import csv
from pprint import pprint
import argparse

data_path_split = '/mnt/data/hand-activity-data/extracted-features/'
data_path_y_split = '/mnt/data/hand-activity-data/split/'

parser = argparse.ArgumentParser(description='')

parser.add_argument('--classifier', type=str, default='rf', help='classifier to use, options: rf, svm, pac')
parser.add_argument('--num_samples', type=int, default=200, help='Number of samples per class for online learning')
parser.add_argument('--classes_to_remove', type=int, default=0, help='Number of classes to remove for online learning')
parser.add_argument('--use_labels', type=bool, default=True, help='Use labels or use predictions of classifier for training')

parsed_args = parser.parse_args()
args = parsed_args.__dict__


def load_data_user_round(u, r, feature_extractor):
    data = np.load(data_path_split + f'extracted_features_for_user{feature_extractor}/svm_features_round{r}_user{u}.npy')
    if len(data) == 0:
        return np.array([], dtype=np.int64).reshape(0, 500)
    return data


def get_classifier():
    classifier_type = args['classifier']
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif classifier_type == 'svm':
        classifier = lm.SGDClassifier(loss='hinge', n_jobs=-1)
    elif classifier_type == 'pac':
        classifier = lm.PassiveAggressiveClassifier(n_jobs=-1)
    else:
        raise Exception('wrong classifier option')
    return classifier


def load_train_data(user):
    train_users = list(range(1, 13))
    train_users.remove(user)
    x_train = np.concatenate([load_data_user_round(u, r, user) for u, r in itertools.product(train_users, range(1, 5))])
    y_train = np.concatenate([np.load(data_path_y_split + f'round{r}_user{u}_Y.npy') for u, r in itertools.product(train_users, range(1, 5))])
    return x_train, y_train


def reduce_classes_of_online_set(x_online, y_online):
    classes_to_keep = list(range(0, 25))
    random.shuffle(classes_to_keep)
    for i in range(args['classes_to_remove']):
        classes_to_keep.pop()

    indexes_to_keep = [y in classes_to_keep for y in y_online]
    x_online = x_online[indexes_to_keep]
    y_online = y_online[indexes_to_keep]
    return x_online, y_online


def load_test_data(user):
    test_rounds = [3]
    online_rounds = [1, 2]

    x_test = np.concatenate([load_data_user_round(user, r, user) for r in test_rounds])
    y_test = np.concatenate([np.load(data_path_y_split + f'round{r}_user{user}_Y.npy') for r in test_rounds])

    x_online = np.concatenate([load_data_user_round(user, r, user) for r in online_rounds])
    y_online = np.concatenate([np.load(data_path_y_split + f'round{r}_user{user}_Y.npy') for r in online_rounds])

    x_online, y_online = reduce_classes_of_online_set(x_online, y_online)

    online_learning_samples = args['num_samples'] * 25
    if online_learning_samples < len(y_online):
        _, x_online, _, y_online = train_test_split(x_online, y_online, test_size=online_learning_samples, stratify=y_online)
    return x_online, y_online, x_test, y_test


accuracies_before_training = []
accuracies = []

for n_feature_extractor in range(1, 13):
    print(f'Testing for user {n_feature_extractor}')
    x_train, y_train_value = load_train_data(n_feature_extractor)
    x_online, y_online, x_test, y_test = load_test_data(n_feature_extractor)

    print('x_train: {}'.format(x_train.shape))
    print('x_online: {}'.format(x_online.shape))
    print('x_test: {}'.format(x_test.shape))

    classifier = get_classifier()
    classifier.fit(x_train, y_train_value)

    # use predictions of model instead of actual labels
    if not args['use_labels']:
        y_online = classifier.predict(x_online)

    accuracy_before_training = classifier.score(x_test, y_test)
    accuracies_before_training.append(accuracy_before_training)
    print('Accuracy on remaining test set: {}'.format(accuracy_before_training))

    if args['classifier'] == 'rf':
        classifier.n_estimators = 700
        classifier.fit(x_online, y_online)
    else:
        best_accuracy = []
        for i in range(1, 10):
            classifier.partial_fit(x_online, y_online)

    accuracy = classifier.score(x_test, y_test)
    accuracy_train = classifier.score(x_online, y_online)
    print('Accuracy: {}, train Accuracy: {}'.format(accuracy, accuracy_train))
    accuracies.append(accuracy)


date = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
experiment_stats = {
    "date": date,
    "classifier": args['classifier'],
    "num_samples": args['num_samples'],
    "labelled": args['use_labels'],
    "lasses_removed": args['classes_to_remove'],
    "accuracy_before": np.average(accuracies_before_training),
    "sd_before": np.std(accuracies_before_training),
    "max_before": np.max(accuracies_before_training),
    "accuracy_after": np.average(accuracies),
    "sd_after": np.std(accuracies),
    "max_after": np.max(accuracies),
    "accuracies_before": accuracies_before_training,
    "accuracies_after": accuracies,
}

pprint(experiment_stats)

filename = f'./runs/{date}.json'
with open(filename, 'w') as file:
    file.write(json.dumps(experiment_stats))

csv_columns = [
    "date",
    "classifier",
    "num_samples",
    "labelled",
    "lasses_removed",
    "accuracy_before",
    "sd_before",
    "max_before",
    "accuracy_after",
    "sd_after",
    "max_after",
    "accuracies_before",
    "accuracies_after",
]

with open('runs/runs.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    # writer.writeheader()
    writer.writerow(experiment_stats)