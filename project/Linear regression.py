import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

with tf.device('/gpu:0'):
    # data load
    # x(input), y(result) data
    dataset_path = ''
    column_names = ['year', 'SO2', 'NO2', 'O3', 'CO', 'pm10', 'rdm']

    raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=",", skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset.tail()
    
    train_dataset = dataset.sample(frac=0.6, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    
    sns.pairplot(train_dataset[["year", "SO2", "NO2", "O3", "CO", "pm10", "rdm"]], diag_kind="kde")
    #plt.savefig('sns_pairplot_result.png')

    train_stats = train_dataset.describe()
    train_stats.pop("year")
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset.pop('rdm')
    test_labels = test_dataset.pop('rdm')

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    def build_model():
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]))
        model.add(Dense(1))
        model.add(Dense(1))

        rms = optimizers.RMSprop(0.001)
        model.compile(loss='mse', optimizer=rms, metrics=['mae', 'mse'])
        return model

    model = build_model()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model.summary()

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    example_result

    class PrintDot(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    EPOCHS = 1000

    hist = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    
    def plot_hist(hist):
        result = pd.DataFrame(hist.history)
        result['epoch'] = hist.epoch

        plt.figure(figsize=(8, 12))

        plt.subplot(2, 1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [rdm]')
        plt.plot(result['epoch'], result['mae'], label='Train Error')
        plt.plot(result['epoch'], result['val_mae'], label='Val Error')
        plt.ylim([0, 5])
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [$rdm^2$]')
        plt.plot(result['epoch'], result['mse'], label='Train Error')
        plt.plot(result['epoch'], result['val_mse'], label='Val Error')
        plt.ylim([0, 20])
        plt.legend()
        #plt.savefig('sns_pairplot_result1.png')
        plt.show()

    plot_hist(hist)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("테스트 세트의 평균 절대 오차: {:5.2f} rdm".format(mae))

    test_predictions = model.predict(normed_test_data).flatten()
    print(normed_test_data)

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [rdm]')
    plt.ylabel('Predictions [rdm]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    #plt.savefig('sns_pairplot_result2.png')
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [rdm]")
    _ = plt.ylabel("Count")
    #plt.savefig('sns_pairplot_result3.png')
    plt.show()

    #export_path=''
    #model.save(export_path, save_format="tf")
