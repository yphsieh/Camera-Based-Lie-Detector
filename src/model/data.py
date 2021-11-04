import pandas as pd
import argparse
import os
import glob
import numpy as np
from itertools import product

OVER_PAD_LENGTH_COUNT = 0


class Subject:
    pulse_dir = 'pulse_dir'  # Can be modified from the command line

    def __init__(self, name, label_df, data_description):
        """
        variables:
            name(subject name)
            pulse_data: a list with 3 members, 0 for general questions, 1 for video 1, 2 for video 2
                        Each member is a list of 20 BPM series
            label_data: an array with shape (3, 20) 
            data_description: option: "pulse"/ "blink"
        """
        global OVER_PAD_LENGTH_COUNT

        self.name = name
        self.pulse_data = []
        self.data_description = data_description

        # reads the pulse of general questions
        self.pulse_data.append(self._read_series_data(
            dir_name=os.path.join(Subject.pulse_dir, name)))

        # reads the pulse of questions for video 1
        self.pulse_data.append(self._read_series_data(
            dir_name=os.path.join(Subject.pulse_dir, name + '_v1')))

        # reads the pulse of questions for video 2
        self.pulse_data.append(self._read_series_data(
            dir_name=os.path.join(Subject.pulse_dir, name + '_v2')))

        # add label for general questions
        self.label_data = np.ones((1, 20)).astype(np.int)
        # add label for questions for video 1
        self.label_data = np.append(
            self.label_data, label_df.iloc[:, 1].values.reshape(1, -1), axis=0)

        self.label_data = np.append(
            self.label_data, label_df.iloc[:, 3].values.reshape(1, -1), axis=0)

        # keep track of the length of each pulse data
        self.pulse_length = self.get_series_data_length()

    def _read_series_data(self, dir_name):
        """
        Reads the BPM/blink csv files. Called in __init__.
        """
        global OVER_PAD_LENGTH_COUNT

        files = glob.glob(dir_name + '/*.csv')
        files.sort()
        if len(files) != 21:
            # print(len(files))
            raise AssertionError(
                'No BPM files or the number of files is wrong in {}: {}'.format(dir_name, len(files)))
        pulse_list = []
        for f in files[1:]:  # Do not read the first file
            if self.data_description == "pulse":
                pulse_series = pd.read_csv(f)['BPM'].values
            elif self.data_description == "blink":
                pulse_series = pd.read_csv(f)['EAR'].values
            else:
                raise AssertionError("No such type of data description")

            pulse_list.append(pulse_series)
        return pulse_list

    def get_series_data_length(self):
        """
        Count the length of each pulse series and save it in "length"
        :return: length
        """
        global OVER_PAD_LENGTH_COUNT

        length = []
        for k in range(3):
            for i in range(len(self.pulse_data[k])):
                length.append(self.pulse_data[k][i].shape[0])
        return length

    def get_pad_data(self, pad_pulse_length):
        """
        Return all data of 60 questions into a numpy array with shape (60, pad_pulse_length)
        If the data is shorter than pulse_length, pad 0 in the end
        If the data is longer, prune off the front
        """
        pad_pulse_data = []
        for k in range(3):
            for i in range(len(self.pulse_data[k])):
                if self.pulse_data[k][i].shape[0] < pad_pulse_length:
                    padded = np.pad(self.pulse_data[k][i],
                                    (pad_pulse_length - self.pulse_data[k][i].shape[0], 0)).reshape(1, -1)
                    # print(padded)
                else:
                    padded = self.pulse_data[k][i][(-1)
                                                   * pad_pulse_length:].reshape(1, -1)
                if k == 0 and i == 0:
                    pad_pulse_data = padded
                else:
                    pad_pulse_data = np.append(pad_pulse_data, padded, axis=0)
                # print(e)

        self.pad_pulse_data = pad_pulse_data
        # print(self.pad_pulse_data.shape)
        return np.array(pad_pulse_data)

    def get_combined_data(self, pad_pulse_length):
        """
        Get a (800, pad_pulse_length, 2) vector. Each data is a (2, pad_pulse_length) vector
        , where a general question is in the first dimension, and a video question is the the second dimension.
        When padding a time series, '0's are add in the end of the series
        If the pulse series is longer, the pulse series would not be used
        :return: all_combined_data, combined_label
        """
        global OVER_PAD_LENGTH_COUNT
        
        all_combined_data = []
        all_combined_label = np.array([])
        for k in range(1, 3):
            for (i, j) in product(range(len(self.pulse_data[0])), range(len(self.pulse_data[k]))):
                if self.pulse_data[0][i].shape[0] > pad_pulse_length or self.pulse_data[k][j].shape[
                        0] > pad_pulse_length:
                    OVER_PAD_LENGTH_COUNT += 1
                    continue
                else:
                    padded_general = np.pad(self.pulse_data[0][i], (0,
                                                                    pad_pulse_length - self.pulse_data[0][i].shape[
                                                                        0])).reshape(1, -1)
                    padded_video = np.pad(self.pulse_data[k][j], (0,
                                                                  pad_pulse_length - self.pulse_data[k][j].shape[
                                                                      0])).reshape(1, -1)
                    combined_data = np.append(padded_general, padded_video, axis=0).T.reshape(
                        1, pad_pulse_length, 2)
                    # print(combined_data.shape)
                    try:
                        all_combined_data = np.append(
                            all_combined_data, combined_data, axis=0)
                    except Exception as e:
                        all_combined_data = combined_data
                        # print(e)

                    combined_label = self.label_data[k][j]
                    all_combined_label = np.append(
                        all_combined_label, combined_label)

        return all_combined_data, all_combined_label

    def __str__(self):
        """
        What happens when print(subject) is called. Overrides the print method.
        """
        print('=' * 20, "Subject Information", '=' * 20)
        print("Subject Name: {}".format(self.name))
        print("Pulse Data Length for general questions")
        print(self.pulse_length[0:20])
        print("Number of general Questions: {}".format(
            len(self.pulse_data[0])))
        print("Pulse Data Length for video 1")
        print("Number of questions for video 1: {}".format(
            len(self.pulse_data[1])))
        print(self.pulse_length[20:40])
        print("Pulse Data Length for video 2")
        print("Number of questions for video 2: {}".format(
            len(self.pulse_data[0])))
        print(self.pulse_length[40:60])
        print('Label Data')
        print(self.label_data)
        print('Label Data shape: {}'.format(self.label_data.shape))

        return ''


def readData(pulse_dir, label_filename, data_description):
    """
    Read the Pulse data with labels. Return a list storing all subjects
    """
    print('=' * 20, "reading in data...", '=' * 20)
    global Subject

    # Modify the root directory of pulse data in Subject
    Subject.pulse_dir = pulse_dir

    # Read the label file
    label_file = pd.read_excel(label_filename)
    subject_names = label_file['subject'].dropna().values

    subject_list = []
    for i in range(len(subject_names)):
        try:
            subject_list.append(
                Subject(subject_names[i], label_df=label_file.iloc[i * 20:(i + 1) * 20:, 1:], data_description=data_description))
        except Exception as e:
            print("Exeception in data.py")
            print(e)

    print('=' * 20, "All valid data are read", '=' * 20)
    return subject_list, subject_names


class SubjectGroup:
    """
    Manipulate all subjects at once
    """

    def __init__(self, pulse_dir, label_filename, data_description="pulse"):
        """
        params:
            data_description: option: "pulse"/ "blink"
        """
        self.subject_list, self.subject_names = readData(
            pulse_dir, label_filename, data_description=data_description)

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, item):
        return self.subject_list[item]

    def get_pad_data(self, pad_pulse_length):
        """
        Combine all padded pulse data to a large numpy array.
        This is the training data and testing data for the auto-encoder.
        :param pad_pulse_length: Length each pulse series is padded to.
        :return: pad_pulse_data (np.array): shape(60*number of subjects, pad_pulse_length)
        """
        pad_pulse_data = []
        if hasattr(self, 'pad_pulse_data'):
            if self.pad_pulse_data.shape[1] == pad_pulse_length:
                return self.pad_pulse_data
        else:
            for i in range(len(self.subject_list)):
                padded = self.subject_list[i].get_pad_pulse_data(
                    pad_pulse_length)
                if i == 0:
                    pad_pulse_data = padded
                else:
                    pad_pulse_data = np.append(pad_pulse_data, padded, axis=0)

        self.pad_pulse_data = pad_pulse_data
        return self.pad_pulse_data

    def get_combined_data(self, pad_pulse_length, exclude_subject_name_list=None):
        """
        The training and testing data for the classifier
        :param pad_pulse_length:
        :param subject_name_list: do not include the data of a subject of it is in exclude_subject_name_list.
                If not specified, all subjects would be used
        :return:
        """
        data, label = [], []
        for i in range(len(self)):
            if exclude_subject_name_list is not None:
                if self.subject_list[i].name in exclude_subject_name_list:
                    continue
            subject_data, subject_label = self.subject_list[i].get_combined_data(
                pad_pulse_length=pad_pulse_length)
            # print(subject_data.shape)
            # print(subject_label.shape)
            try:
                data = np.append(data, subject_data, axis=0)
                label = np.append(label, subject_label, axis=0)
            except Exception as e:
                data = subject_data
                label = subject_label
                print("Exception in data.py")
                print(e)
        # print(label)
        label[label == 0] = 2
        label = label - 1
        # print(label)
        return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    parser.add_argument('-dir', '--data_dir',
                        default='../data_pulse', help='directory to the pulse data')
    parser.add_argument('-dd', '--data_description', default="pulse",
                        help="which data to use: \"pulse\" or \"blink\"")
    args = parser.parse_args()

    print(args)

    subject_group = SubjectGroup(
        pulse_dir=args.data_dir, label_filename='../label.xlsx', data_description=args.data_description)
    # data = subject_group.get_pad_pulse_data(pad_pulse_length=500)
    print(subject_group.subject_names)

    #data, label = subject_group.get_combined_data(600, exclude_subject_name_list=['hzlbm', 'zyt_sheep', 'qiduo0818'])
    # print(data.shape)
    # print(label.shape)
    #np.save('data/data_600.npy', data)
    #np.save('data/label_600.npy', label)

    data, label = subject_group.get_combined_data(600)
    print(data.shape)
    print(label.shape)
    np.save('data/data_all_600_{}.npy'.format(args.data_description), data)
    np.save('data/label_all_600_{}.npy'.format(args.data_description), label)
    print("OVER PAD LENGTH COUNT: {}".format(OVER_PAD_LENGTH_COUNT))

    # data, label = subject_group.get_combined_data(600, exclude_subject_name_list=[
    #                                             'ban9975', 'dingyiyi', 'family', 'erik', 'liyunjie', '8888ba_one', 'leo', 'hsinyu', 'gingerbread', 'cgmc_0410', 'eason_1029'])
    #print(data.shape)
    #print(label.shape)
    #np.save('data/data_test_600.npy', data)
    #np.save('data/label_test_600.npy', label)
