import numpy as np
import pandas as pd
import neurokit2 as nk
import scipy.integrate
import scipy.stats
# from pickle_to_csv import * # DON'T uncomment (would cause a loop)

EDA = 'EDA'
ACC = 'ACC'
TEMP = 'TEMP'
ECG = 'ECG'
EMG = 'EMG'
RESP = 'Resp'
BVP = 'BVP'
SCL = 'SCL'
SCR = 'SCR'

def relative_integral(signal, start, stop, sampling_rate):
  return scipy.integrate.simps(signal.iloc[start:stop].to_numpy() - signal.iloc[start]) / sampling_rate

def create_getter(func, func_name):
  def getter(signal, name, axis=None):
    axis_names = {0: 'x', 1: 'y', 2: 'z'}
    if axis is None:
      value = func(signal)
      name = '{}_{}'.format(name, func_name)
    else:
      value = func(signal[:,axis])
      axis_name = axis_names[axis] if axis in axis_names else 'axis_{}'.format(axis)
      name = '{}_{}_{}'.format(name, axis_name, func_name)
    return {'{}_{}'.format(name, func_name): func(signal)}
  return getter

get_mean = create_getter(np.mean, 'mean')
get_std = create_getter(np.mean, 'std')
get_min = create_getter(np.min, 'min')
get_min = create_getter(np.max, 'max')
get_rms = create_getter(lambda s: np.sqrt(np.mean(s**2)), 'root_mean_square')
get_slope = create_getter(lambda s: (s[-1] - s[0]) / 100, 'slope') # NOTE assuming 100 second window
# get_dynamic_range = create_getter(lambda s: np.log10(np.max(s) / np.min(s)), 'dynamic_range') TODO (maybe don't log?)
get_absolute_integral = create_getter(lambda s: scipy.integrate.simps(np.abs(s)), 'absolute_integral') # integrate the absolute value

def get_all(signal, name, getters, axes=None):
  features = {}
  for getter in getters:
    if axes is None:
      features.update(getter(signal, name))
    else:
      for axis in axes:
        features.update(getter(signal, name, axis=axis))
  return features

## FEATURE GETTERS. Each takes in a signal (raw or a processed data frame), baseline info (perhaps nothing), and sampling rate (perhaps meaningless)
def get_temp_features(raw_temp, baseline_info, sampling_rate):
  temp_getters = [get_mean, get_std, get_min, get_max, get_slope,] # get_dynamic_range] FIXME include?
  return get_all(raw_temp, TEMP, temp_getters)

def get_acc_features(raw_acc, baseline_info, sampling_rate):
  acc_magnitudes = np.array([np.sqrt(x**2 + y**2 + z**2) for x,y,z in raw_acc])
  acc_features = get_all(raw_acc, ACC, [get_mean, get_std, get_absolute_integral, get_min, get_max], axes=[0,1,2])
  acc_features.update(get_all(acc_magnitudes, ACC + '_magnitude', [get_mean, get_std, get_absolute_integral]))
  return acc_features

# eda, _ = standardized_eda_process(eda_signal, rate)
def get_eda_features(eda, baseline_info, sampling_rate): # df with columns EDA_Standardized, EDA_Tonic (SCL), EDA_Phasic (SCR), 
  eda_getters = [get_mean, get_std, get_min, get_max, get_slope,] #get_dynamic_range] FIXME include?
  eda_features = get_all(eda.EDA_Standardized, EDA, eda_getters)
  eda_features.update(get_all(eda.EDA_Tonic, SCL, [get_mean, get_std]))
  eda_features.update(get_all(eda.EDA_Phasic, SCR, [get_std])) # FIXME include mean??
  eda_features[SCR + '_num_segments'] = sum(eda.SCR_Onsets)
  onsets = eda[eda.SCR_Onsets == 1.0]
  peaks = eda[eda.SCR_Peaks == 1.0]
  eda_features[SCR + '_sum_startle_magnitudes'] = sum(peaks.SCR_Amplitude)
  eda_features[SCR + '_sum_response_durations'] = sum(peaks.SCR_RiseTime)
  eda_features[SCR + '_response_area'] = sum([relative_integral(eda.EDA_Phasic, onset, peak, sampling_rate) for onset, peak in zip(onsets.index, peaks.index)])
  eda_features[SCL + '_time_correlation'] = scipy.stats.pearsonr(eda.EDA_Tonic, eda.index)[0]
  return eda_features

# resp, _ = nk.rsp_process(raw_resp, sampling_rate=rate)
def get_resp_features(resp, baseline_info, sampling_rate):
  peaks = list(resp.index[resp.RSP_Peaks == 1]) # could also intersect info with current indices
  troughs = list(resp.index[resp.RSP_Troughs == 1])
  start = resp.index[0]
  end = resp.index[-1]
  start_inhaling = troughs[0] > peaks[0]
  end_inhaling = troughs[-1] > peaks[-1]
  inhale_zip = lambda: zip(peaks + [end] if end_inhaling else peaks, [start] + troughs if start_inhaling else troughs)
  exhale_zip = lambda: zip(peaks if start_inhaling else [start] + peaks, troughs if end_inhaling else troughs + [end])
  inhale_durations = [(peak - trough) / sampling_rate for peak, trough in inhale_zip()]
  exhale_durations = [(trough - peak) / sampling_rate for peak, trough in exhale_zip()]
  volume = sum([relative_integral(resp.RSP_Clean, trough, peak, sampling_rate) for peak, trough in inhale_zip()])
  
  features = get_all(inhale_durations, RESP + '_inhale', [get_mean, get_std])
  features.update(get_all(inhale_durations, RESP + '_exhale', [get_mean, get_std]))
  features[RESP + '_inhale_exhale_ratio'] = sum(inhale_durations) / sum(exhale_durations)
  features[RESP + '_volume'] = volume
  features[RESP + '_breath_rate'] = resp.RSP_Rate.iloc[-1]
  # TODO stretch and resp duration?
  return features

def my_hrv(peaks, sampling_rate):
  result = []
  result.append(nk.hrv_time(peaks, sampling_rate=sampling_rate))
  result.append(nk.hrv_frequency(peaks, sampling_rate=sampling_rate, vlf=(0.01, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4), vhf=(0.4, 1)))
  return pd.concat(result, axis=1)

def get_hrv_features(peaks, sampling_rate):
  hrv = my_hrv(peaks, sampling_rate)
  renamings = {
    'HRV_VLF': 'HRV_ultra_low_freq',
    'HRV_LF': 'HRV_low_freq',
    'HRV_HF': 'HRV_high_freq',
    'HRV_VHF': 'HRV_ultra_high_freq',
    'HRV_LFn': 'HRV_low_freq_normalized', 
    'HRV_HFn': 'HRV_high_freq_normalized',
    'HRV_LFHF': 'HRV_low_high_freq_ratio',
    'HRV_MeanNN': 'HRV_mean',
    'HRV_SDNN': 'HRV_std',
    'HRV_RMSSD': 'HRV_rms',
    'HRV_pNN50': 'HRV_percent_large_intervals',
    'HRV_TINN': 'HRV_tinn'
  }
  hrv = hrv[renamings.keys()]
  frequency_columns = ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFn', 'HRV_HFn']
  hrv[frequency_columns] = hrv[frequency_columns].fillna(0.0)
  hrv['HRV_LFHF'] = hrv['HRV_LFHF'].fillna(1.0)
  hrv = hrv.rename(columns=renamings, errors='raise')
  return {col: val for col, val in zip(hrv.columns, hrv.to_numpy()[0])}

# ecg, _ = nk.ecg_process(ecg, sampling_rate=rate)
def get_ecg_features(ecg, baseline_info, sampling_rate):
  features = get_all(ecg.ECG_Rate, ECG, [])
  features.update(get_hrv_features(ecg.ECG_R_Peaks))
  return features

# bvp, _ = nk.ppg_process(bvp, sampling_rate=rate)
def get_bvp_features(bvp, baseline_info, sampling_rate):
  features = get_all(bvp.BVP_Rate, BVP, [])
  features.update(get_hrv_features(bvp.PPG_Peaks))
  return features

## IN PROCESSS features
def get_emg_features(emg, baseline_info, sampling_rate):
  pass # TODO

## OLD STUFF
# make sure these are the same as in the pickle_to_csv.py

def get_signal_features(chest_sigs_and_rates, wrist_sigs_and_rates):
  result = {name: np.max(sig_and_rate[0]) for name, sig_and_rate in chest_sigs_and_rates.items()}
  result.update({name: np.max(sig_and_rate[0]) for name, sig_and_rate in wrist_sigs_and_rates.items()})
  return result
  # FIXME calculate real features for all signals
