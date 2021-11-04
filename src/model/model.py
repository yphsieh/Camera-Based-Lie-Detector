from keras import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Bidirectional, LeakyReLU

def construct_model():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(600,2)))
    model.add(LSTM(units=256, return_sequences=True, input_shape=(600, 4)))
    model.add(Bidirectional(LSTM(units=256, dropout=0.2,
                                 recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=256, dropout=0.2, recurrent_dropout=0.2)))

    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()
    return model

# other things to do: leaky relu, sigmoid, add more neurons
# other things to do: change class weights
# cross validation
# use mcc score
# ensemble with blink/ each other


"""
model:
    ./save/RDNN_s22_06-10 19:42:26.h5
    ./save/RDNN_s22_06-10 20:11:11.h5
    ./save/RDNN_s22_06-10 20:35:50.h5
    
model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(600, 2)))
model.add(Bidirectional(LSTM(units=256, dropout=0.2,
                                recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(units=256, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(units=256))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(units=128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()
"""

"""
model:
    ./save/RDNN_s22_06-10 18:33:55.h5

model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(600, 2)))
model.add(Bidirectional(LSTM(units=256, dropout=0.2,
                                recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(units=256, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(units=256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()
"""

"""
model: 
    ./save/RDNN_s22_06-10 17:15:45.h5
    ./save/RDNN_s22_06-10 18:00:55.h5

model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(600, 2)))
model.add(Bidirectional(LSTM(units=256, dropout=0.2,
                                recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(units=256, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()
"""