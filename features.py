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
RESP = 'RESP'
BVP = 'BVP'
SCL = 'SCL'
SCR = 'SCR'

def make_name(prefix, for_wrist):
  return ('wrist_' if for_wrist else 'chest_') + prefix

def relative_integral(signal, start, stop, sampling_rate):
  return scipy.integrate.simps(signal.iloc[start:stop].to_numpy() - signal.iloc[start]) / sampling_rate

def create_getter(func, func_name):
  def getter(signal, name, axis=None):
    axis_names = {0: 'x', 1: 'y', 2: 'z'}
    if axis is None:
      display_name = '{}_{}'.format(name, func_name)
      return {display_name: func(signal)} 
    axis_name = axis_names[axis] if axis in axis_names else 'axis_{}'.format(axis)
    display_name = '{}_{}_{}'.format(name, axis_name, func_name)
    return {display_name: func(signal[:,axis])}
  return getter

get_mean = create_getter(np.mean, 'mean')
get_std = create_getter(np.mean, 'std')
get_min = create_getter(np.min, 'min')
get_max = create_getter(np.max, 'max')
get_rms = create_getter(lambda s: np.sqrt(np.mean(s**2)), 'root_mean_square')
get_slope = create_getter(lambda s: (s[-1] - s[0]) / 60, 'slope') # NOTE assuming 60 second window
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
def get_temp_features(raw_temp, for_wrist, baseline_info, sampling_rate):
  temp_getters = [get_mean, get_std, get_min, get_max, get_slope,] # get_dynamic_range] FIXME include?
  return get_all(raw_temp, make_name(TEMP, for_wrist), temp_getters)

def get_acc_features(raw_acc, for_wrist, baseline_info, sampling_rate):
  acc_magnitudes = np.array([np.sqrt(x**2 + y**2 + z**2) for x,y,z in raw_acc])
  acc_features = get_all(raw_acc, make_name(ACC, for_wrist), [get_mean, get_std, get_absolute_integral, get_min, get_max], axes=[0,1,2])
  acc_features.update(get_all(acc_magnitudes, make_name(ACC + '_magnitude', for_wrist), [get_mean, get_std, get_absolute_integral]))
  return acc_features

 # protected against no peaks
def get_eda_features(eda, for_wrist, baseline_info, sampling_rate): # df with columns EDA_Standardized, EDA_Tonic (SCL), EDA_Phasic (SCR), 
  eda_getters = [get_mean, get_std, get_min, get_max, get_slope,] #get_dynamic_range] FIXME include?
  eda_features = get_all(eda.EDA_Standardized.to_numpy(), make_name(EDA, for_wrist), eda_getters)
  eda_features.update(get_all(eda.EDA_Tonic, make_name(SCL, for_wrist), [get_mean, get_std]))
  eda_features.update(get_all(eda.EDA_Phasic, make_name(SCR, for_wrist), [get_std])) # FIXME include mean??
  eda_features[make_name(SCL + '_time_correlation', for_wrist)] = scipy.stats.pearsonr(eda.EDA_Tonic, eda.index)[0]

  num_onsets = sum(eda.SCR_Onsets)
  num_peaks = sum(eda.SCR_Peaks)
  onsets = eda[eda.SCR_Onsets == 1.0]
  peaks = eda[eda.SCR_Peaks == 1.0]
  if num_onsets == 0 or num_peaks == 0 or num_peaks == 1 and peaks.index[0] < onsets.index[0]:
    for name in ['num_segments', 'sum_startle_magnitudes', 'sum_response_durations', 'response_area']:
      eda_features[make_name('{}_{}'.format(SCR, name), for_wrist)] = 0
  else:
    eda_features[make_name(SCR + '_num_segments', for_wrist)] = num_onsets
    eda_features[make_name(SCR + '_sum_startle_magnitudes', for_wrist)] = sum(peaks.SCR_Amplitude)
    eda_features[make_name(SCR + '_sum_response_durations', for_wrist)] = sum(peaks.SCR_RiseTime)
    peak_start_index = 1 if peaks.index[0] < onsets.index[0] else 0
    eda_features[make_name(SCR + '_response_area', for_wrist)] = sum([relative_integral(eda.EDA_Phasic, onset, peak, sampling_rate) for onset, peak in zip(onsets.index, peaks.index[peak_start_index:])])
  return eda_features

 # NOT protected against no peaks
def get_resp_features(resp, for_wrist, baseline_info, sampling_rate):
  peaks = list(resp.index[resp.RSP_Peaks == 1]) # could also intersect info with current indices
  troughs = list(resp.index[resp.RSP_Troughs == 1])
  start = resp.index[0]
  end = resp.index[-1]
  start_inhaling = troughs[0] > peaks[0]
  end_inhaling = troughs[-1] > peaks[-1]

  inhale_zip = zip(peaks[1:] if start_inhaling else peaks, troughs)
  inhale_durations = [(peak - trough) / sampling_rate for peak, trough in inhale_zip]

  exhale_zip = zip(peaks, troughs[1:] if start_inhaling else troughs)
  exhale_durations = [(trough - peak) / sampling_rate for peak, trough in exhale_zip]

  inhale_volume_zip = zip(peaks + [end] if end_inhaling else peaks, [start] + troughs if start_inhaling else troughs)
  volume = sum([relative_integral(resp.RSP_Clean, trough, peak, sampling_rate) for peak, trough in inhale_volume_zip if peak - trough > 3]) 

  features = get_all(inhale_durations, make_name(RESP + '_inhale', for_wrist), [get_mean, get_std])
  features.update(get_all(exhale_durations, make_name(RESP + '_exhale', for_wrist), [get_mean, get_std]))
  features[make_name(RESP + '_inhale_exhale_ratio', for_wrist)] = sum(inhale_durations) / sum(exhale_durations)
  features[make_name(RESP + '_volume', for_wrist)] = volume / 60 # FIXME change if window size isn't 60 seconds
  features[make_name(RESP + '_breath_rate', for_wrist)] = resp.RSP_Rate.iloc[-1]
  # TODO stretch and resp duration?
  return features

def my_hrv(peaks, sampling_rate):
  result = []
  result.append(nk.hrv_time(peaks, sampling_rate=sampling_rate))
  result.append(nk.hrv_frequency(peaks, sampling_rate=sampling_rate, vlf=(0.01, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4), vhf=(0.4, 1)))
  return pd.concat(result, axis=1)

all_renamings = {}

def get_hrv_features(peaks, prefix, sampling_rate):
  hrv = my_hrv(peaks, sampling_rate)
  if prefix not in all_renamings:
    all_renamings[prefix] = {
      'HRV_VLF': prefix + 'HRV_ultra_low_freq',
      'HRV_LF': prefix + 'HRV_low_freq',
      'HRV_HF': prefix + 'HRV_high_freq',
      'HRV_VHF': prefix + 'HRV_ultra_high_freq',
      'HRV_LFn': prefix + 'HRV_low_freq_normalized', 
      'HRV_HFn': prefix + 'HRV_high_freq_normalized',
      'HRV_LFHF': prefix + 'HRV_low_high_freq_ratio',
      'HRV_MeanNN': prefix + 'HRV_mean',
      'HRV_SDNN': prefix + 'HRV_std',
      'HRV_RMSSD': prefix + 'HRV_rms',
      'HRV_pNN50': prefix + 'HRV_percent_large_intervals',
      'HRV_TINN': prefix + 'HRV_tinn'
    }
  renamings = all_renamings[prefix]
  hrv = hrv[renamings.keys()]
  frequency_columns = ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFn', 'HRV_HFn']
  hrv[frequency_columns] = hrv[frequency_columns].fillna(0.0)
  hrv['HRV_LFHF'] = hrv['HRV_LFHF'].fillna(1.0)
  hrv = hrv.rename(columns=renamings, errors='raise')
  return {col: val for col, val in zip(hrv.columns, hrv.to_numpy()[0])}

def get_ecg_features(ecg, for_wrist, baseline_info, sampling_rate):
  features = get_all(ecg.ECG_Rate, make_name('ECG_HR', for_wrist), [get_mean, get_std])
  features.update(get_hrv_features(ecg.ECG_R_Peaks, make_name('ECG_', for_wrist), sampling_rate))
  return features

def get_bvp_features(bvp, for_wrist, baseline_info, sampling_rate):
  features = get_all(bvp.PPG_Rate, make_name('BVP_HR', for_wrist), [get_mean, get_std])
  features.update(get_hrv_features(bvp.PPG_Peaks, make_name('BVP_', for_wrist), sampling_rate))
  return features

## IN PROCESSS features
def get_emg_features(emg, for_wrist, baseline_info, sampling_rate):
  pass # TODO
