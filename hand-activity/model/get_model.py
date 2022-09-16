from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model, load_model

# model_path = "/home/sysgen/workspace/hand-activity-data/models/model_hand-activities.h5"
model_path = "models/model_hand-activities.h5"


def get_model(args):
    if args['model'] == 'laput':
        model = get_model_laput(args)
    elif args['model'] == 'vgg16':
        model = get_model_VGG16(args)
    elif args['model'] == 'ours':
        model = get_our_model(args)
    else:
        raise ValueError('Bad parameter for model.')
    return model


def get_model_VGG16(args):
    # TODO: replace this model as it does not perform particularly well, check out efficientNet
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(256, 48, 3)))

    headModel = baseModel.output
    # TODO: classification part does not seem to be optimal
    # headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(26, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    opt = Adam(lr=args['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model


def get_model_laput(args):
    model = load_model(model_path)
    print("model loaded")
    # clone model to reinitialize weights
    model = clone_model(model)
    opt = Adam(lr=args['learning_rate'])
    model.compile(opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def get_our_model(args):
    # Input: dimensions 256 x 48 x 3
    # 1. Conv Unit (depth 64): 3 x 3, Batch Norm, ReLu, Max Pool (stride 2 x 2)
    # 2. Conv Unit (depth 128): 3 x 3, Batch Norm, ReLu, Max Pool (stride 2 x 2)
    # 3. Conv Unit (depth 256): 3 x 3, Batch Norm, ReLu, Max Pool (stride 2 x 2)
    # 4. Conv Unit (depth 512): 3 x 3, Batch Norm, ReLu, Max Pool (stride 2 x 2)
    # 5. Conv Unit (depth 1024): 3 x 3, Batch Norm, ReLu, Max Pool (stride 2 x 2)
    # 1. Fully Connected: n = 2000, Batch Norm, ReLu
    # 2. Fully Connected: n = 500, Batch Norm, ReLu
    # Dropout p = 0.4
    # Softmax: n = 25

    model = Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(256, 48, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(1024, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(units=2000))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(Dense(units=500))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='features'))
    model.add(Dropout(0.40))
    model.add(Dense(units=26, activation="softmax"))

    opt = Adam(lr=args['learning_rate'])
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])
    model.summary()
    return model
