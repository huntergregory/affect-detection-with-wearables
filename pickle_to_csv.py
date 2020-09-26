import numpy as np
import pandas as pd
import math
from features import get_signal_features

EDA = 'EDA'
ACC = 'ACC'
TEMP = 'TEMP'
ECG = 'ECG'
EMG = 'EMG'
RESP = 'Resp'
BVP = 'BVP'

shared_measurements = [ACC, TEMP, EDA] # 32, 4, and 4 Hz for wrist
chest_measurements = [ECG, EMG, RESP] + shared_measurements # all at 700 Hz
wrist_measurements = [BVP] + shared_measurements # 64 Hz

WRIST_SAMPLE_RATES = {
  BVP: 64,
  ACC: 32,
  TEMP: 4,
  EDA: 4,
}

# don't worry about these
POSSIBLE_SAMPLE_RATES = [2, 4, 7, 10, 14, 20, 25, 28, 35, 50, 70]
assert(all([700/rate == 700//rate for rate in POSSIBLE_SAMPLE_RATES]))
DIVISORS = {rate: 700//rate for rate in POSSIBLE_SAMPLE_RATES}
AVG_DOWNSAMPLE = 'avg'
POINT_DOWNSAMPLE = 'point'

## MODIFY THESE. All window sizes in seconds
OUT_FILE = '/work/hlg16/WESAD.csv'
WESAD_FOLDER = '../../WESAD/WESAD/'
subject_ids = [4] # [k for k in range(2, 18) if k != 12]
BASIC_WINDOW_SIZE = 60
WINDOW_SIZES = {
  ECG: BASIC_WINDOW_SIZE,
  BVP: BASIC_WINDOW_SIZE,
  EDA: BASIC_WINDOW_SIZE,
  EMG: BASIC_WINDOW_SIZE,
  RESP: BASIC_WINDOW_SIZE,
  TEMP: BASIC_WINDOW_SIZE,
  ACC: 5,
}
MAX_SIZE = max(WINDOW_SIZES.values())
SKIP_LENGTH = MAX_SIZE # seconds to skip between observation

# don't use this
DOWNSAMPLING_INFO = {
  ECG: (28, POINT_DOWNSAMPLE), # FIXME?
  BVP: (28, POINT_DOWNSAMPLE), # FIXME doesn't go into 64,
  EDA: (7, AVG_DOWNSAMPLE), # FIXME?
  EMG: (7, POINT_DOWNSAMPLE), # FIXME?
  RESP: (7, POINT_DOWNSAMPLE),
  TEMP: (4, AVG_DOWNSAMPLE),
  ACC: (7, POINT_DOWNSAMPLE),
}

def downsample(values, length_at_700_hz, sampling):
  new_rate, kind = sampling
  divisor = DIVISORS[new_rate]
  new_length = length_at_700_hz // divisor 
  if new_length == len(values):
    return values
  if kind == POINT_DOWNSAMPLE:
    return values[::divisor]
  if kind == AVG_DOWNSAMPLE:
    return [np.mean(values[k*divisor:((k+1)*divisor)]) for k in range(new_length)]
  raise ValueError('Unknown downsampling kind')

## CREATING THE CSV
filenames = [WESAD_FOLDER + 'S{}/S{}.pkl'.format(k, k) for k in subject_ids]
all_subjects = None
for filename in filenames:
  # logistics for loading data
  data = pd.read_pickle(filename)
  print('loaded data for ' + filename)
  labels = data['label']
  chest = data['signal']['chest']
  chest[TEMP] = chest['Temp']
  wrist = data['signal']['wrist']
  for m in wrist_measurements:
    if m != ACC:
      wrist[m] = wrist[m].reshape(-1)
  for m in chest_measurements:
    if m != ACC:
      wrist[m] = chest[m].reshape(-1)

  # creating a DataFrame for the subject
  df = pd.DataFrame()
  length_at_700_hz = len(data['label'])
  total_time = length_at_700_hz / 700
  time = MAX_SIZE
  count = 0
  approx_final_count = total_time // SKIP_LENGTH
  while time < total_time:
    count += 1
    if count % (approx_final_count // 10)  == 0:
      print('added {} out of about {} rows so far'.format(count, approx_final_count))
    # make sure all observations in this time frame are the same label
    end_chest_index = 700 * time - 1
    similar = 0
    for k in range(700 * MAX_SIZE):
      k = end_chest_index - k
      if labels[end_chest_index] != labels[k]:
        break
      similar += 1
    if similar != 700 * MAX_SIZE:
      good_time = math.ceil(similar / 700)
      time += MAX_SIZE - good_time
      print('labels are not consistent. Skipping ahead with similar count of {} and good time of {}'.format(similar, good_time))
      continue

    # compute features
    def get_signal(name, is_wrist):
      signal_data = chest; sample_rate = 700
      if is_wrist:
        sample_rate = WRIST_SAMPLE_RATES[name]
        signal_data = wrist
      size = WINDOW_SIZES[name]
      end = sample_rate * time
      start = sample_rate * (time - size)
      return signal_data[name][start:end]
    
    chest_signals = {measurement: get_signal(measurement, False) for measurement in chest_measurements}
    wrist_signals = {measurement: get_signal(measurement, True) for measurement in wrist_measurements}
    
    new_row = {'subject_id': data['subject']}
    new_row.update(get_signal_features(chest_signals, wrist_signals))
    df.append(new_row, ignore_index=True)
    time += SKIP_LENGTH


  if all_subjects is None:
    all_subjects = df
  else:
    all_subjects.append(df)
  print()

all_subjects.to_csv(OUT_FILE)
