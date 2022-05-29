import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from keras import optimizers
from tensorflow.python.keras.models import load_model
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# GPU 사용
with tf.device('/gpu:0'):
    # Chart 기본 크기 설정
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False
    
    # 데이터 불러오기
    dataset_path = '사용한 데이터 파일'
    column_names = ['Date Time', 'SO2', 'NO2', 'O3', 'CO', 'pm10', 'death']

    raw_df = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=",", skipinitialspace=True) #, index_col=0

    df = raw_df.copy()

    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%Y%m')

    # 산점도 행렬
    #df.index = df['Date']
    #df = df.drop(['Date'], axis=1)
    #print(df)
    #df.plot(subplots=True)
    #plt.show()

    # 훈련시킬 데이터 범위 설정 및 재현성 보장을 위한 시드 설정
    TRAIN_SPLIT = 120
    tf.random.set_seed(13)
    
    # 데이터 셋 정규화(표준화)
    # 다중 특성을 사용한 모델 학습
    features_considered = ['SO2', 'NO2', 'O3', 'CO', 'pm10', 'death']
    features = df[features_considered]
    features.index = df['Date Time']
    #features.plot(subplots=True)
    #plt.show()

    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    # 데이터 셋 분할(train, validation)
    def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])

        return np.array(data), np.array(labels)

    # past = 학습할 과거 데이터 양, target = 예측할 미래 데이터양
    past_history = 60
    future_target = 12
    STEP = 6

    # 예측하고 싶은 데이터 선택 - dataset[:, @]
    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 5], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)

    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 5],
                                                       TRAIN_SPLIT, None, past_history,
                                                       future_target, STEP,
                                                       single_step=True)

    print('Single window of past history : {}'.format(x_train_single[0].shape))

    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    print(x_train_single.shape[-2:])

    def build_single_step_model():
        model = Sequential()
        model.add(LSTM(32, input_shape=x_train_single.shape[-2:]))
        model.add(Dense(1))

        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
        return model

    #for x, y in val_data_single.take(1):
    #    print(model.predict(x).shape)

    def train_single_step_model():
        EVALUATION_INTERVAL = 100
        EPOCHS = 10

        train = model.fit(train_data_single, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_data_single, validation_steps=50)
        return train

    def save_model():
        model.save('모델을 저장할 폴더')

    model = build_single_step_model()
    model.summary()

    hist = train_single_step_model()
    #save_model()
    #model = load_model('모델이 저장된 폴더')

    # Single-step 예측
    def plot_train_history(history, title):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(loss))

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(title)
        plt.legend()
        plt.show()

    plot_train_history(hist, 'Single Step Training and validation loss')

    # 모델 테스트
    for x, y in val_data_single.take(3):
        plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                          model.predict(x)[0]], 12,
                         'Single Step Prediction')
