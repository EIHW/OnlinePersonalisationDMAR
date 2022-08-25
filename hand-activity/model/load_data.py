import numpy as np

data_path = '/mnt/data/hand-activity-data/'
# data_path = '/home/sysgen/workspace/hand-activity-data/'


def load_data(args, i):
    if args['type'] == 'per-user':
        return load_data_per_user_accuracy(i)
    elif args['type'] == 'leave-user-out':
        return load_data_leave_user_out(i)
    elif args['type'] == 'all-user':
        return load_data_all_user_accuracy(i)
    elif args['type'] == 'post-removal':
        return load_data_accuracy_post_removal(i)
    else:
        raise ValueError('Bad parameter for type.')


def load_data_per_user_accuracy(user):
    round1_x = np.load(data_path + 'split/round1_user{}_X.npy'.format(user))
    round1_y = np.load(data_path + 'split/round1_user{}_Y.npy'.format(user))

    round2_x = np.load(data_path + 'split/round2_user{}_X.npy'.format(user))
    round2_y = np.load(data_path + 'split/round2_user{}_Y.npy'.format(user))

    if len(round1_x) == 0:
        x_train = round2_x
        y_train = round2_y
    elif len(round2_x) == 0:
        x_train = round1_x
        y_train = round1_y
    else:
        x_train = np.concatenate([round1_x, round2_x])
        y_train = np.concatenate([round1_y, round2_y])

    x_eval = np.load(data_path + 'split/round3_user{}_X.npy'.format(user))
    y_eval = np.load(data_path + 'split/round3_user{}_Y.npy'.format(user))

    return x_train, y_train, x_eval, y_eval


def load_data_leave_user_out(user):
    train_users = list(range(1, 13))
    train_users.remove(user)

    x_train = np.concatenate([np.load(data_path + 'users/features/U{}_features_X.npy'.format(u)) for u in train_users])
    y_train = np.concatenate([np.load(data_path + 'users/features/U{}_features_Y.npy'.format(u)) for u in train_users])

    x_eval = np.load(data_path + 'users/features/U{}_features_X.npy'.format(user))
    y_eval = np.load(data_path + 'users/features/U{}_features_Y.npy'.format(user))

    return x_train, y_train, x_eval, y_eval


def load_data_all_user_accuracy(round):
    train_rounds = list(range(1, 5))
    train_rounds.remove(round)

    x_train = np.concatenate([np.load(data_path + 'rounds/round{}_features_X.npy'.format(r)) for r in train_rounds])
    y_train = np.concatenate([np.load(data_path + 'rounds/round{}_features_Y.npy'.format(r)) for r in train_rounds])

    x_eval = np.load(data_path + 'rounds/round{}_features_X.npy'.format(round))
    y_eval = np.load(data_path + 'rounds/round{}_features_Y.npy'.format(round))

    return x_train, y_train, x_eval, y_eval


def load_data_accuracy_post_removal(user):
    x_train, y_train, _, _ = load_data_per_user_accuracy(user)

    x_eval = np.load(data_path + 'split/round4_user{}_X.npy'.format(user))
    y_eval = np.load(data_path + 'split/round4_user{}_Y.npy'.format(user))

    return x_train, y_train, x_eval, y_eval
