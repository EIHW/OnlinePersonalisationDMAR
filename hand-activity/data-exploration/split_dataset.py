import numpy as np

# TODO:
data_path = '/mnt/data/hand-activity-data/'
save_path = './restructured_data/'


def load_data_by_user(user):
    x = np.load(data_path + 'users/features/U{}_features_X.npy'.format(user))
    y = np.load(data_path + 'users/features/U{}_features_Y.npy'.format(user))
    labels = np.load(data_path + 'users/features/U{}_features_labels.npy'.format(user))
    return x, y, labels


def includes(arr, val):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (high + low) // 2

        if arr[mid] < val:
            low = mid + 1
        elif arr[mid] > val:
            high = mid - 1
        else:
            return True
    return False


def get_round(spectrogram):
    hashed_value = hash(bytes(spectrogram))

    if includes(round1, hashed_value):
        return 1
    if includes(round2, hashed_value):
        return 2
    if includes(round3, hashed_value):
        return 3
    if includes(round4, hashed_value):
        return 4

    raise ValueError('could not determine round of spectrogram.')


# loading the data of the different rounds. Hash and sort them for faster searching.
round1 = np.load(data_path + 'rounds/round1_features_X.npy')
round1 = np.array([hash(bytes(v)) for v in round1])
round1 = np.sort(round1)

round2 = np.load(data_path + 'rounds/round2_features_X.npy')
round2 = np.array([hash(bytes(v)) for v in round2])
round2 = np.sort(round2)

round3 = np.load(data_path + 'rounds/round3_features_X.npy')
round3 = np.array([hash(bytes(v)) for v in round3])
round3 = np.sort(round3)

round4 = np.load(data_path + 'rounds/round4_features_X.npy')
round4 = np.array([hash(bytes(v)) for v in round4])
round4 = np.sort(round4)

datapoints_not_found = 0

# idea: iterate through the users, check in which round sample was recorded and safe it accordingly.
for user in range(1, 13):
    print('starting for user {}'.format(user))
    X, Y, labels = load_data_by_user(user)
    data_with_round = []
    # first add information about the round
    for index, value in enumerate(X):
        try:
            r = get_round(value)
            data_with_round.append((r, user, value, Y[index], labels[index]))
        except ValueError:
            datapoints_not_found += 1
            continue

    # then iterate through rounds and select relevant data to save
    for searched_round in range(1, 5):
        print('saving data for round {}'.format(searched_round))
        relevant_data_x = []
        relevant_data_y = []
        relevant_data_labels = []
        for (r, u, x, y, label) in data_with_round:
            if r == searched_round:
                relevant_data_x.append(x)
                relevant_data_y.append(y)
                relevant_data_labels.append(label)
        save_x = np.array(relevant_data_x)
        save_y = np.array(relevant_data_y)
        save_labels = np.array(relevant_data_labels)

        print('saving {} samples'.format(len(save_x)))

        np.save(save_path + 'round{}_user{}_X.npy'.format(searched_round, user), save_x)
        np.save(save_path + 'round{}_user{}_Y.npy'.format(searched_round, user), save_y)
        np.save(save_path + 'round{}_user{}_labels.npy'.format(searched_round, user), save_labels)

print('Could not find a total of {} datapoints'.format(datapoints_not_found))
