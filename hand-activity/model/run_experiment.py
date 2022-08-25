'''
The purpose of this script is to replicate the results of Laput and Harrison.
'''
import os
import argparse
import numpy as np
import json
from shutil import copyfile
from datetime import datetime
from pprint import pprint

from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import TensorBoard

from load_data import load_data
from preprocessing import extract_features, extract_features_restructured, encode_labels
from get_model import get_model


parser = argparse.ArgumentParser(description='')

parser.add_argument('--type', type=str, default='per-user', help='type of experiment: per-user, all-user, post-removal or leave-user-out')
parser.add_argument('--model', type=str, default='laput', help='model to use, options: laput, vgg16, ours')
parser.add_argument('--encoding', type=str, default='fixed', help='encoding of the spectrograms: original or fixed')
parser.add_argument('--gpu_number', type=int, default=0, help='GPU to use: 0-3 (Default: 0)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (Default: 32)')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--experiment_name', type=str, default='', help='Name of the experiment')
parser.add_argument('--epochs', type=int, default=50, help='Number of Epochs to train')

parsed_args = parser.parse_args()
args = parsed_args.__dict__

os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_number'])


def save_results(model, log_dir, accuracy):
    """
    Saves all the relevant data for each run of the experiment:
    * script
    * finished model

    :param model: the fully trained keras model
    :return:
    """
    save_model(model, log_dir + '/model.h5', save_format='h5')
    copyfile('./run_experiment.py', log_dir + '/run_experiment.py')
    copyfile('./get_model.py', log_dir + '/get_model.py')
    copyfile('./load_data.py', log_dir + '/load_data.py')
    with open(log_dir + '/parameters.json', 'w') as f:
        json.dump(args, f)
    with open(log_dir + '/accuracy', 'w') as f:
        f.write(str(accuracy))


if args['type'] == 'all-user':
    iterations = range(1, 5)
else:
    iterations = range(1, 13)

# data
categorical_accuracies = []

for i in iterations:
    print('Iteration: {}'.format(i))
    model = get_model(args)
    model.summary()

    x_train, y_train_value, x_val, y_val_value = load_data(args, i)
    y_train = encode_labels(y_train_value)
    y_val = encode_labels(y_val_value)
    if args['encoding'] == 'fixed':
        x_train = extract_features_restructured(x_train)
        x_val = extract_features_restructured(x_val)
    else:
        x_train = extract_features(x_train)
        x_val = extract_features(x_val)

    # tensorboard
    log_dir = "runs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + args['type'] + '_' + args['experiment_name'] + '_' + str(i)
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # checkpointer = ModelCheckpoint(log_dir + 'best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # stopper = EarlyStopping(monitor='val_loss', mode='min', patience=10)

    for layer in model.layers:
        layer.trainable = True

    print("[INFO] training ...")
    print("experiment settings: \n")
    pprint(args)
    H = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=args['epochs'],
        batch_size=args['batch_size'],
        callbacks=[tensorboard_callback] #, checkpointer, stopper]
    )

    # feature_extractor = get_feature_extractor(model)
    # train_features_for_svm = extract_features_for_svm(feature_extractor, x_train)
    # test_features_for_svm = extract_features_for_svm(feature_extractor, x_val)
    # train_svm(train_features_for_svm, y_train_value, test_features_for_svm, y_val_value)

    accuracy = H.history['val_categorical_accuracy'][-1]
    categorical_accuracies.append(accuracy)
    save_results(model, log_dir, accuracy)

# store avg accuracy over different rounds/users
overall_results = {
    "accuracy": str(np.average(categorical_accuracies)),
    "sd": str(np.std(categorical_accuracies)),
    "max": str(np.max(categorical_accuracies)),
}

with open(log_dir + '/total_accuracy', 'w') as f:
    json.dump(overall_results, f)
