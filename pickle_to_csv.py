import numpy as np
import pandas as pd
import math
import neurokit2 as nk
from measurement import Measurement
from features import get_temp_features, get_acc_features, get_eda_features, get_resp_features, get_emg_features, get_ecg_features, get_bvp_features

## Special Process Functions
no_process = lambda signal, rate: (signal, None)

def standardized_eda_process(eda_signal, sampling_rate, method="neurokit"):
  eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate, method=method)
  eda_cleaned_standardized = nk.standardize(eda_cleaned) # only change to eda_process
  eda_decomposed = nk.eda_phasic(eda_cleaned_standardized, sampling_rate=sampling_rate)
  peak_signal, info = nk.eda_peaks(eda_decomposed["EDA_Phasic"].values, sampling_rate=sampling_rate, method=method, amplitude_min=0.1)
  signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned, "EDA_Standardized": eda_cleaned_standardized})
  signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)
  return signals, info

def my_emg_process(emg_signal, sampling_rate):
  return nk.emg_process(emg_signal, sample_rating=sampling_rate) # TODO update

## Baseline Info Functions
no_baseline_info = lambda baseline, rate: None

############################
## MODIFY THESE. All window sizes in seconds
EDA = 'EDA'
ACC = 'ACC'
TEMP = 'TEMP'
ECG = 'ECG'
EMG = 'EMG'
RESP = 'RESP'
BVP = 'BVP'

OUT_FILE = '/work/hlg16/WESAD-one-sec-shift.csv'
WESAD_FOLDER = '../../WESAD/WESAD/'
subject_ids = [k for k in range(2, 18) if k != 12]
BASIC_WINDOW_SIZE = 60 # a couple dependencies on this in features.py
EDA_UPSCALE_FACTOR = 2
MEASUREMENTS = {
  ECG: Measurement(BASIC_WINDOW_SIZE, no_baseline_info, nk.ecg_process, get_ecg_features, has_chest_data=True),
  BVP: Measurement(BASIC_WINDOW_SIZE, no_baseline_info, nk.ppg_process, get_bvp_features, wrist_rate=64),
  EDA: Measurement(BASIC_WINDOW_SIZE, no_baseline_info, standardized_eda_process, get_eda_features, has_chest_data=True, wrist_rate=4, wrist_upscale=EDA_UPSCALE_FACTOR),
  # EMG: Measurement(BASIC_WINDOW_SIZE, my_emg_process, get_emg_features, has_chest_data=True), FIXME uncomment
  RESP: Measurement(BASIC_WINDOW_SIZE, no_baseline_info, nk.rsp_process, get_resp_features, has_chest_data=True),
  TEMP: Measurement(BASIC_WINDOW_SIZE, no_baseline_info, no_process, get_temp_features, has_chest_data=True, wrist_rate=4),
  ACC: Measurement(5, no_baseline_info, no_process, get_acc_features, has_chest_data=True, wrist_rate=4)
}
SKIP_LENGTH = 1 # seconds to skip between observation
############################

BASELINE_LABEL = 1
AMUSEMENT_LABEL = 3
STRESS_LABEL = 2
MAX_SIZE = max([m.window_size for m in MEASUREMENTS.values()])
filenames = [WESAD_FOLDER + 'S{}/S{}.pkl'.format(k, k) for k in subject_ids]
all_subjects = None

filenames = ['../../WESAD/WESAD/S11/S11.pkl']  # FIXME remove
for filename in filenames: 
  data = pd.read_pickle(filename)
  print('loaded data for ' + filename)
  chest = data['signal']['chest']
  chest[TEMP] = chest['Temp']
  chest[RESP] = chest['Resp']
  wrist = data['signal']['wrist']

  label = -1
  changes = []
  for k in range(len(data['label'])):
    curr_label = data['label'][k]
    if curr_label != label:
      changes.append((k, curr_label))
    label = curr_label
      
  def get_times(target_label):
    end = len(data['label'])
    for j, index_label in enumerate(changes):
      if index_label[1] == target_label:
        break
    start = int(math.ceil(changes[j][0] / 700))
    end = int(math.floor(changes[j+1][0] / 700)) # if j+1 < len(changes) else int(len(data['label']) / 700)
    return start, end

  baseline_times = get_times(BASELINE_LABEL)
  amusement_times = get_times(AMUSEMENT_LABEL)
  stress_times = get_times(STRESS_LABEL)

  print('baseline times: {}'.format(baseline_times))
  print('amusement times: {}'.format(amusement_times))
  print('stress times: {}'.format(stress_times))

  for name, measurement in MEASUREMENTS.items():
      if measurement.has_wrist_data:
        print('processing raw {} wrist data'.format(name))
        measurement.process_raw(wrist[name], True, baseline_times, amusement_times, stress_times)
      if measurement.has_chest_data:
        print('processing raw {} chest data'.format(name))
        measurement.process_raw(chest[name], False, baseline_times, amusement_times, stress_times)

  # make a DataFrame for the subject
  df = pd.DataFrame()
  def update_df(for_amusement):
    start, end = amusement_times if for_amusement else stress_times
    time = MAX_SIZE
    while time <= end - start:
      if time % (end - start) // 10 == 0:
        print('getting features for time {}'.format(time))
      new_row = {'subject_id': data['subject'], 'affect': 'amusement' if for_amusement else 'stress'}
      for measurement in MEASUREMENTS.values():
        new_row.update(measurement.get_features(time, for_amusement))
      global df
      df = df.append(new_row, ignore_index=True)
      time += SKIP_LENGTH

  update_df(True)
  update_df(False)

  if all_subjects is None:
    all_subjects = df
  else:
    all_subjects = all_subjects.append(df)
  print()

print('final df info')
print('len: ', len(all_subjects))
print('columns: ', all_subjects.columns)
print(all_subjects.describe())
all_subjects.to_csv(OUT_FILE)