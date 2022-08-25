import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Nadam, RMSprop
from tensorflow.keras import backend as K

filter_size = 5
num_filters = 64
num_classes = 12
learning_rate = 0.00001
dropout_rate = 0.5


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_model():
    model = Sequential()
    model.add(layers.Input((100, 40)))
    # model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.MaxPool1D(2))

    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.MaxPool1D(2))

    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.MaxPool1D(2))

    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    # model.add(layers.Dense(512, activation='relu', name='features'))
    # model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))

    opt = Nadam(lr=learning_rate)
    # opt = RMSprop(learning_rate=1e-2, decay=0.95)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy', f1_m,precision_m, recall_m])
    model.summary()
    return model


# def load_model3():
#     model = Sequential()
#     model.add(layers.Input((100, 40, 1)))
#
#     model.add(layers.Conv2D(num_filters, kernel_size=(5, 1), strides=(1, 1)))
#     model.add(layers.ReLU())
#     model.add(layers.BatchNormalization())
#     model.add(layers.Conv2D(num_filters, kernel_size=(5, 1), strides=(1, 1)))
#     model.add(layers.ReLU())
#     model.add(layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1)))
#
#     model.add(layers.Conv2D(num_filters, kernel_size=(5, 1), strides=(1, 1)))
#     model.add(layers.ReLU())
#     model.add(layers.Conv2D(num_filters, kernel_size=(5, 1), strides=(1, 1)))
#     model.add(layers.ReLU())
#     model.add(layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1)))
#
#     model.add(layers.Dropout(dropout_rate))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(128))
#     model.add(layers.ReLU())
#
#     model.add(layers.Dropout(dropout_rate))
#     model.add(layers.Dense(128))
#     model.add(layers.ReLU())
#
#     model.add(layers.Dense(num_classes, activation='softmax'))
#
#     opt = Adam(lr=learning_rate)
#     # opt = RMSprop(learning_rate=0.001, rho=0.95)
#     model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])
#     model.summary()
#     return model


def load_model2():
    model = Sequential()
    model.add(layers.Input((100, 40)))

    model.add(keras.layers.Conv1D(filters=256, kernel_size=5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv1D(filters=256, kernel_size=5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv1D(filters=256, kernel_size=5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    opt = Nadam(lr=learning_rate)
    # opt = RMSprop(learning_rate=0.001, rho=0.95)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])
    model.summary()
    return model
