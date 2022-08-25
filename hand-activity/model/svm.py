from sklearn import svm, metrics
from tensorflow.keras.models import Model
import numpy as np


def get_feature_extractor(model):
    extraction_model = Model(inputs=model.input, outputs=model.get_layer('activation_6').output)
    return extraction_model


def extract_features_for_svm(feature_extractor, input_data):
    features = feature_extractor.predict(input_data)
    return features


def train_svm(features, labels, features_eval, labels_eval):
    # TODO: take part of training set instead of eval set to find best complexity, this is not fully correct
    complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

    # TODO: find out if we have to scale the results first and if so how does that influence online training on device
    results = []

    for comp in complexities:
        # print('\nComplexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0, max_iter=10000)
        clf.fit(features, labels)

        y_pred = clf.predict(features_eval)
        print(y_pred)
        print(labels_eval)
        acc = metrics.accuracy_score(labels_eval, y_pred)
        results.append(acc)

    optimum_complexity = complexities[np.argmax(results)]
    print('\nSVM - Optimum complexity: {0:.6f}, maximum categorical accuracy on Devel {1:.1f}\n'.format(optimum_complexity, np.max(results)))

    clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
    clf.fit(features, labels)
    return clf
