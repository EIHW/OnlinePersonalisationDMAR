import pandas as pd
import numpy as np

path = 'data/pamap2/PAMAP2_Dataset'

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

columns = [
    'timestamp',
    'activityID',
    'heart_rate',
    # IMU - HAND
    'imu_hand-temp',
    'imu_hand-acc_x',
    'imu_hand-acc_y',
    'imu_hand-acc_z',
    '_1',
    '_2',
    '_3',
    'imu_hand-gyro_x',
    'imu_hand-gyro_y',
    'imu_hand-gyro_z',
    'imu_hand-magneto_x',
    'imu_hand-magneto_y',
    'imu_hand-magneto_z',
    '_4',
    '_5',
    '_6',
    '_26',
    # IMU - chest
    'imu_chest-temp',
    'imu_chest-acc_x',
    'imu_chest-acc_y',
    'imu_chest-acc_z',
    '_7',
    '_8',
    '_9',
    'imu_chest-gyro_x',
    'imu_chest-gyro_y',
    'imu_chest-gyro_z',
    'imu_chest-magneto_x',
    'imu_chest-magneto_y',
    'imu_chest-magneto_z',
    '_10',
    '_11',
    '_12',
    '_22',
    # IMU - ankle
    'imu_ankle-temp',
    'imu_ankle-acc_x',
    'imu_ankle-acc_y',
    'imu_ankle-acc_z',
    '_13',
    '_14',
    '_15',
    'imu_ankle-gyro_x',
    'imu_ankle-gyro_y',
    'imu_ankle-gyro_z',
    'imu_ankle-magneto_x',
    'imu_ankle-magneto_y',
    'imu_ankle-magneto_z',
    '_16',
    '_17',
    '_18',
    '_28',
]


def load_data_for_user(user):
    df = pd.read_csv(f'{path}/Protocol/subject10{user}.dat', sep=' ', names=columns)
    df = df.drop(columns=['_4', '_5', '_6', '_26', '_10', '_11', '_12', '_22', '_16', '_17', '_18', '_28'])
    # df = df.drop(columns=['_1', '_2', '_3', '_4', '_5', '_6', '_26', '_7', '_8', '_9', '_10', '_11', '_12', '_22',
    #                       '_13', '_14', '_15', '_16', '_17', '_18', '_28'])
    # df = df.drop(columns=['heart_rate', 'imu_hand-temp', 'imu_chest-temp', 'imu_ankle-temp'])
    # print(df.isna().sum())
    # exit(0)
    df = df.fillna(method='bfill')
    return df


def split_data_into_activities(df):
    data_per_activity = {}
    label = 0
    for i in activities:
        df1 = df[df['activityID'] == i]
        data_per_activity[label] = df1
        label += 1
    return data_per_activity


def rolling_frame(df, length, hop):
    # one row equals 10ms
    start = 0
    end = len(df)

    frame_begin = start
    frame_end = start + length
    samples = []
    while frame_end < end:
        df1 = df[frame_begin:frame_end]
        samples.append(df1)
        frame_begin = frame_begin + hop
        frame_end = frame_end + hop
    return samples


def extract_features(df):
    df = df.drop(columns=['timestamp', 'activityID'])
    return df.to_numpy().swapaxes(0, 1)


def create_dataset_for_user(user):
    df = load_data_for_user(user)
    data_per_activity = split_data_into_activities(df)

    x = []
    y = []
    for key, data in data_per_activity.items():
        frames = rolling_frame(data, 300, 30)
        x = x + [extract_features(frame) for frame in frames]
        y = y + [key for i in frames]

    return np.array(x), np.array(y)


def leave_user_out_data(user):
    users = list(range(1, 10))
    users.remove(user)

    x_train = np.array([], dtype=np.float64).reshape(0, 40, 300)
    y_train = np.array([], dtype=np.int64).reshape(0)

    for u in users:
        user_x, user_y = create_dataset_for_user(u)
        x_train = np.concatenate([x_train, user_x])
        y_train = np.concatenate([y_train, user_y])

    x_test, y_test = create_dataset_for_user(user)

    return x_train, y_train, x_test, y_test
