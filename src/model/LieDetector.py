import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import keras.backend as K
from keras import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.metrics import Recall
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score

from data_preprocessing import *
from sklearn.utils import class_weight
import evaluation as ev
import model as md

from itertools import combinations
from scipy.special import comb


def get_f1(y_true, y_pred):  # taken from old keras source code
    """
    Define f1 score for tensors
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


parser = argparse.ArgumentParser(description='')
# actions
parser.add_argument('--train',  action='store_true', default=False)
parser.add_argument('--test',   action='store_true', default=False)
parser.add_argument('-m', '--model_name', required=False, type=str, help="the model to load when not train")
# parameters for data preprocessing
parser.add_argument('--seed',   type=int,   default=22)
parser.add_argument('--smooth', action='store_true',  default=False)
parser.add_argument('--scale',  action='store_true',  default=False)
# model metric
parser.add_argument('--metric', type=str, default="f1score")
# class weights
parser.add_argument('-cw', '--class_weights', nargs=2, type=float,
                    required=False, help="the weight of class 0 and 1 (ex: 0.5, 10)")
# randomly choose 2 subjects as validation data
parser.add_argument('-bv', '--better_validation', action='store_true',
                    default=False, help="the better way to split train and validation")
parser.add_argument('-bvi', '--better_validation_subjects_index', type=int, required=False, help="Manually specifies the subjects chosen as validation")
# directory
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./save')
args = parser.parse_args()
print(args)

np.random.seed(args.seed)


# Split data into train and val
if args.better_validation:
    # Choosing 3 subjects as validation
    if args.better_validation_subjects_index is None:
        x_train, x_val, y_train, y_val = SplitTrainVal(0.2, args.seed, better_validation=args.better_validation)
    else:
        # Get the validaton subject list based on "args.better_validation_subjects_index"
        val_sub_list = list(combinations(subject_name_list, 3))[ int(args.better_validation_subjects_index % comb(len(subject_name_list), 3)) ]
        x_train, x_val, y_train, y_val = SplitTrainVal(0.2, args.seed, better_validation=args.better_validation, val_sub_list=val_sub_list)
else:
    # Splitting data after merging them all first
    data = np.load(os.path.join(args.data_dir, 'data_600.npy'))
    label = np.load(os.path.join(args.data_dir, 'label_600.npy')).reshape(-1, 1)
    x_train, x_val, y_train, y_val = SplitTrainVal(0.2, args.seed, data=data, label=label)


# Smooth/ Scale data
x_train, x_val, y_train, y_val = DataPreprocess(x_train, x_val, y_train, y_val, rdm=args.seed, smooth=args.smooth, scale=args.scale)


current_time = str(datetime.datetime.now())[5:-7]
model_path = os.path.join(args.save_dir, 'RDNN_s{}_{}.h5'.format(
    args.seed, current_time))
# current time is append to model path
print("Model Path: {}".format(model_path))

model = md.construct_model()
if args.train:

    if args.class_weights is None:
        class_weights = dict(enumerate(class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train.reshape(-1))))
    else:
        class_weights = dict(enumerate(args.class_weights))

    print(class_weights)

    if args.metric == "f1score":
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy', Recall(), get_f1])

        callbacks = []
        callbacks.append(ModelCheckpoint(
            model_path, monitor='val_get_f1', verbose=1, save_best_only=True, save_weights_only=True,  mode='max'))

        csv_logger = CSVLogger(os.path.join(
            args.save_dir, 'RDNN_log_s{}_{}.csv'.format(args.seed, current_time)), separator=',', append=False)
        callbacks.append(csv_logger)

        earlystop = EarlyStopping(monitor='val_get_f1', patience=5, mode='max')
        callbacks.append(earlystop)

        tStart = time.time()
        model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val),
                  class_weight=class_weights, shuffle=True, batch_size=256, callbacks=callbacks)
        print('Costing time: ', time.time()-tStart, ' ......')

    elif args.metric == "accuracy":
        # accuracy
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = []
        callbacks.append(ModelCheckpoint(
            model_path, monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max'))
        csv_logger = CSVLogger(os.path.join(
            args.save_dir, 'RDNN_log_acc_s{}_{}.csv'.format(args.seed, current_time)), separator=',', append=False)
        callbacks.append(csv_logger)
        earlystop = EarlyStopping(monitor='accuracy', patience=5, mode='max')
        callbacks.append(earlystop)

        tStart = time.time()
        model.fit(x_train, y_train, epochs=100, validation_data=(
            x_val, y_val), class_weight=class_weights, shuffle=True, batch_size=256, callbacks=callbacks)
        print('Costing time: ', time.time()-tStart, ' ......')
else:
    # if args.train is False, the model name must be specified
    assert args.model_name is not None

if args.test:
    if args.model_name is not None:
        print('='*20, 'Model Loading...', '='*20)
        model.load_weights(args.model_name)
        print('='*20, 'Model Loaded', '='*20)

    # probability values
    train_predict = model.predict(x_train)
    val_predict =  model.predict(x_val)

    print("Calculating ROC curve...")
    # ROC curves
    plt.clf()
    fpr, tpr, thresholds = roc_curve(y_train, train_predict)
    plt.plot(fpr, tpr, label="train")
    fpr, tpr, thresholds = roc_curve(y_val, val_predict)
    plt.plot(fpr, tpr, label="val")
    plt.legend()
    if args.model_name is None:
        plt.savefig("ROC_curves/RDNN_s{}_{}_ROC.png".format(args.seed, current_time))
        print("ROC curve saved to ROC_curves/RDNN_s{}_{}_ROC.png".format(args.seed, current_time))
    else:
        plt.savefig("ROC_curves/{}_ROC.png".format(args.model_name[6:-3]))
        print("ROC curve saved to ROC_curves/{}.png".format(args.model_name[6:-3]))
    # AUC scores
    score_auc_train = roc_auc_score(y_train, train_predict)
    score_auc_val = roc_auc_score(y_val, val_predict)

    with open("train_results_ROC.csv", 'a') as f:
        f.write("{}, train, {}".format(model_path, score_auc_train))
        if args.better_validation_subjects_index is not None:
            f.write(",Valid({}) + bvi({})".format(" ".join(val_sub_list), args.better_validation_subjects_index))
        f.write('\n')

        f.write("{}, val, {}".format(model_path, score_auc_val))
        f.write('\n')

    """
    # testing train can val data in order to find the best thres
    for thres in [0.4, 0.48, 0.49, 0.5, 0.51, 0.52, 0.6]:

        # evaluating x_train and y_train
        # y_train_pred: {0,1}
        y_train_pred = (train_predict > thres).astype(np.int)
        score_f1, score_recall, score_acc, score_confusion_matrix = ev.get_scores(
            y_train, y_train_pred)

        print("Writing to csv with thres {}".format(thres))
        with open("train_results.csv", 'a') as f:
            f.write('{},train,{},{},{},{},{}'.format(model_path, score_f1, score_recall,
                                                     score_acc, ' '.join([str(x) for x in score_confusion_matrix.reshape(-1)]), thres))
            
            if args.better_validation_subjects_index is not None and thres==0.4:
                f.write(",Valid({})".format(" ".join(val_sub_list)))
            f.write('\n')

        # evaluating x_val and y_val
       
        y_val_pred = (val_predict > thres).astype(np.int)
        score_f1, score_recall, score_acc, score_confusion_matrix = ev.get_scores(
            y_val, y_val_pred)

        print("Writing to csv with thres {}".format(thres))
        with open("train_results.csv", 'a') as f:
            f.write('{},val,{},{},{},{},{}'.format(model_path, score_f1, score_recall,
                                                    score_acc, ' '.join([str(x) for x in score_confusion_matrix.reshape(-1)]), thres))
            f.write('\n')

    """

