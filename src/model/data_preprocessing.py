import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data as dt

subject_name_list = ['ban9975', 'dingyiyi', 'family', 'erik', 'liyunjie', '8888ba_one', 'leo', 'hsinyu', 'gingerbread', 'cgmc_0410', 'eason_1029', 'hzlbm', 'zyt_sheep', 'qiduo0818']

def SplitTrainVal(ts, rdm, data=None, label=None, better_validation=False, val_sub_list=None):
    """
    Split train and val data
    params:
        ts: test data proportional size
        rdm: seed
        data: All Training + Validation data. Can be unspecified if better_validation is True.
        label: All Training + Validation label. Can be unspecified if better_validation is True.
        better_validation: If set True, then choose 3 random people as validation
        val_sub_list: The list of subject names used as validation. If not specified, randomly pick 3 subjects.
    """
    global subject_name_list

    if better_validation:
        subject_group_pulse = dt.SubjectGroup(pulse_dir='../data_pulse', label_filename='../label.xlsx', data_description="pulse")
        subject_group_blink = dt.SubjectGroup(pulse_dir='../data_blink', label_filename='../label.xlsx', data_description="blink")

        if val_sub_list is None:
            val_subject_name = np.random.choice(subject_name_list, 3, replace=False)
        else:
            val_subject_name = val_sub_list

        print("Choosing {} as validation data".format(" ".join(val_subject_name)))

        # Getting pulse and blink data
        x_train_pulse, y_train_pulse = subject_group_pulse.get_combined_data(600, exclude_subject_name_list=[x for x in val_subject_name])
        x_train_blink, y_train_blink = subject_group_blink.get_combined_data(600, exclude_subject_name_list=[x for x in val_subject_name])
        # Combine pulse and blink data
        x_train = np.concatenate((x_train_pulse, x_train_blink), axis=2)
        assert np.sum(y_train_pulse!= y_train_blink) == 0
        y_train = y_train_pulse

        x_val_pulse, y_val_pulse = subject_group_pulse.get_combined_data(600, exclude_subject_name_list=[x for x in subject_name_list if x not in val_subject_name])
        x_val_blink, y_val_blink = subject_group_blink.get_combined_data(600, exclude_subject_name_list=[x for x in subject_name_list if x not in val_subject_name])
        x_val = np.concatenate((x_val_pulse, x_val_blink), axis=2)
        assert np.sum(y_val_pulse!= y_val_blink) == 0
        y_val = y_val_pulse

    else:
        # Make sure data and label is loaded as np.array already
        assert (data is not None and label is not None)
        x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=ts, random_state=rdm)

    print('x_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('x_val shape: {}'.format(x_val.shape))
    print('y_val shape: {}'.format(y_val.shape))

    lie_ratio_train = np.sum(y_train)/y_train.shape[0]
    print('Lie Ratio Train: {}'.format(lie_ratio_train))
    lie_ratio_val = np.sum(y_val)/y_val.shape[0]
    print('Lie Ratio Val: {}'.format(lie_ratio_val))

    return x_train, y_train, x_val, y_val


def DataPreprocess(x_train, y_train, x_val, y_val, rdm, smooth, scale):
    """
    Scale/ Smooth train and val data
    """
    x_axis = np.linspace(0, len(x_train[0]), len(x_train[0]))
    if not scale:
        plt.plot(x_axis, x_train[rdm*10])

    if smooth:
        for i in range(len(x_train)):
            tmp = [x_train[i][j][0]
                   for j in range(len(x_train[0])) if x_train[i][j][0] != 0]
            tmp = pd.DataFrame(tmp)
            tmp = tmp.ewm(span=300).mean()
            for j in range(len(tmp.values)):
                x_train[i][j][0] = tmp.values[j]

            tmp = [x_train[i][j][1]
                   for j in range(len(x_train[0])) if x_train[i][j][1] != 0]
            tmp = pd.DataFrame(tmp)
            tmp = tmp.ewm(span=300).mean()
            for j in range(len(tmp.values)):
                x_train[i][j][1] = tmp.values[j]

        for i in range(len(x_val)):
            tmp = [x_val[i][j][0]
                   for j in range(len(x_val[0])) if x_val[i][j][0] != 0]
            tmp = pd.DataFrame(tmp)
            tmp = tmp.ewm(span=300).mean()
            for j in range(len(tmp.values)):
                x_val[i][j][0] = tmp.values[j]

            tmp = [x_val[i][j][1]
                   for j in range(len(x_val[0])) if x_val[i][j][1] != 0]
            tmp = pd.DataFrame(tmp)
            tmp = tmp.ewm(span=300).mean()
            for j in range(len(tmp.values)):
                x_val[i][j][1] = tmp.values[j]

    if scale:
        for i in range(len(x_train)):
            ave_pulse = [x_train[i][j][0]
                   for j in range(len(x_train[0])) if x_train[i][j][0] != 0]
            ave_pulse = np.average(ave_pulse)
            x_train[i][:,:2] /= ave_pulse

            ave_blink = [x_train[i][j][2]
                   for j in range(len(x_train[2])) if x_train[i][j][2] != 0]
            ave_blink = np.average(ave_blink)
            x_train[i][:,2:4] /= ave_blink

        for i in range(len(x_val)):
            ave_pulse = [x_val[i][j][0]
                   for j in range(len(x_val[0])) if x_val[i][j][0] != 0]
            ave_pulse = np.average(ave_pulse)
            x_val[i][:,:2] /= ave_pulse

            ave_blink = [x_val[i][j][2]
                   for j in range(len(x_val[2])) if x_val[i][j][2] != 0]
            ave_blink = np.average(ave_blink)
            x_val[i][:,2:4] /= ave_blink

    plt.plot(x_axis, x_train[rdm*10])
    """
    if not scale:
        plt.legend(labels=['original general', 'original data',
                           'processed general', 'processed data'])
    else:
        plt.legend(labels=['processed general', 'processed data'])

    plt.savefig('temp.png')
    """

    return x_train, x_val, y_train, y_val


def TestPreprocess(x_test, smooth, scale):
    if smooth:
        for i in range(len(x_test)):
            tmp = pd.DataFrame(x_test[i])
            tmp.ewm(span=300).mean()
            x_test[i] = tmp.values

    if scale:
        for i in range(len(x_test)):
            ave = [x_test[i][j][0]
                   for j in range(len(x_test[0])) if x_test[i][j][0] != 0]
            ave = np.average(ave)
            x_test[i] /= ave

    return x_test

if __name__ == '__main__':
    x_train, x_val, y_train, y_val = SplitTrainVal(0.2, rdm=22, better_validation=True)
    print(x_train)
    print(x_val)
    x_train, x_val, y_train, y_val = DataPreprocess(x_train, x_val, y_train, y_val, rdm=22, smooth=False, scale=True)
    print(x_train)
    print(x_val)