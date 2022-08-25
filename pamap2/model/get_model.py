import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam

filter_size = 5
num_filters = 64
num_classes = 12
learning_rate = 0.0001
dropout_rate = 0.5


def load_model():
    model = Sequential()
    model.add(layers.Input((300, 40)))

    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.MaxPool1D(2))

    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.MaxPool1D(2))

    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.MaxPool1D(2))

    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.Conv1D(filters=num_filters, kernel_size=filter_size, padding='valid', activation='relu'))
    model.add(layers.MaxPool1D(2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(512, activation='relu', name='features'))
    model.add(layers.Dropout(dropout_rate))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))

    opt = Adam(lr=learning_rate)
    # opt = RMSprop(learning_rate=0.001, rho=0.95)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])
    model.summary()
    return model
