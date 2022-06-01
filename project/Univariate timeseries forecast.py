import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Chart 기본 크기 설정
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# 데이터 불러오기
dataset_path = ''
column_names = ['Date Time', 'SO2', 'NO2', 'O3', 'CO', 'pm10', 'death']

raw_df = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=",", skipinitialspace=True) #, index_col=0

df = raw_df.copy()
date_time = pd.to_datetime(df.pop('Date Time'), format='%Y%d')
#print(df)

# 데이터 시각화
#plot_cols = ['SO2', 'NO2', 'O3', 'CO', 'pm10', 'death']
#plot_features = df[plot_cols]
#plot_features.index = date_time
#_ = plot_features.plot(subplots=True)
#plt.show()

# 데이터 통계
#print(df.describe().transpose())

# 시간 데이터 변환
#timestamp_s = date_time.map(datetime.datetime.timestamp)

#month = 12*24*60*60 #  * H * M * S
#year = (365.2425)*month

#df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
#df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
#df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
#df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

#plt.plot(np.array(df['Month sin'])[:12])
#plt.plot(np.array(df['Month cos'])[:12])
#plt.xlabel('Time [m]')
#plt.title('Time of month signal')
#plt.show()

#fft = tf.signal.rfft(df['pm10'])
#f_per_dataset = np.arange(0, len(fft))

#n_samples_m = len(df['pm10'])
#months_per_year = 12
#years_per_dataset = n_samples_m/(months_per_year)
#print(years_per_dataset)

#f_per_year = f_per_dataset/years_per_dataset
#plt.step(f_per_year, np.abs(fft))
#plt.xscale('log')
#plt.ylim(0, 10000)
#plt.xlim([0.1, max(plt.xlim())])
#plt.xticks([1, 12], labels=['1/Year', '1/month'])
#_ = plt.xlabel('Frequency (log scale)')
#plt.show()

# 데이터 분할
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

# 데이터 정규화
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# 데이터 특성 분포 확인
#df_std = (df - train_mean) / train_std
#df_std = df_std.melt(var_name='Column', value_name='Normalized')
#plt.figure(figsize=(12, 6))
#ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
#_ = ax.set_xticklabels(df.keys(), rotation=90)
#plt.show()

# 데이터 창 작업
class WindowGenerator():
  def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

w1 = WindowGenerator(input_width=12, label_width=1, shift=12, label_columns=['death'])
w1

w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['death'])
w2

# 분할
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[10:10+w2.total_window_size]),
                           np.array(train_df[20:20+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')

# 플롯
w2.example = example_inputs, example_labels

def plot(self, model=None, plot_col='pm10', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [m]')

WindowGenerator.plot = plot

w2.plot(plot_col='death')
plt.show()

# tf.data.Dataset 만들기
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Each element is an (inputs, label) pair
w2.train.element_spec

for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
