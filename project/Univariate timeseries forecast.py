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

    # 데이터 셋 분할(train, validation)
    def univariate_data(dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i+target_size])
        return np.array(data), np.array(labels)

    # 훈련시킬 데이터 범위 설정 및 재현성 보장을 위한 시드 설정
    TRAIN_SPLIT = 120
    tf.random.set_seed(13)

    # 단일 특성만 사용한 모델 학습
    uni_data = df['death']
    uni_data.index = df['Date Time']
    uni_data = uni_data.values

    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data-uni_train_mean)/uni_train_std

    univariate_past_history = 20
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)

    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                               univariate_past_history,
                                               univariate_future_target)

    #print('Sigle window of past history')
    #print(x_train_uni[0])
    #print('\n Target temperature to predict')
    #print(y_train_uni[0])

    # 제공되는 정보 = 파란색, 붉은 X = 값 예측
    def create_time_steps(length):
        return list(range(-length, 0))

    def show_plot(plot_data, delta, title):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i].flatten(), marker[i], label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future+5)*2])
        plt.xlabel('Time-Step')
        plt.show()
        return plt

    #show_plot([x_train_uni[0], y_train_uni[0]], 0, "Sample Example")

    # Baseline - 모델 학습 전 기준치 설정
    # 입력 지점이 주어지면 모든 기록을 보고 다음 지점이 마지막 20개 관측치의
    # 평균이 될 것을 예측
    def baseline(history):
        return np.mean(history)

    #show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, "Baseline Prediction Example")

    # RNN을 사용한 Baseline 예측 비교
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    def build_model():
        model = Sequential()
        model.add(LSTM(8, input_shape=x_train_uni.shape[-2:]))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mae')
        return model

    def train_model():
        EVALUATION_INTERVAL = 100
        EPOCHS = 10
        
        train = model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50)
        return train
    

    def save_model():
        model.save('모델을 저장할 폴더')

    
    model = build_model()
    model.summary()
    
    hist = train_model()
    #save_model()
    #model = load_model('모델이 저장된 폴더')

    # 모델 테스트
    for x, y in val_univariate.take(3):
        plot = show_plot([x[0].numpy(), y[0].numpy(), hist.predict(x)[0]], 0, "Simple LSTM model")
    
