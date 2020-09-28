import numpy as np
import pandas as pd

class Measurement:
  def __init__(self, window_size, baseline_info_function, process_function, feature_getter, has_chest_data=False, wrist_rate=None, wrist_upscale=None):
    self.window_size = window_size
    self.baseline_info_function = baseline_info_function
    self.process_function = lambda signal, rate: process_function(signal, rate)[0] # get the df but leave behind the info
    self.feature_getter = feature_getter
    self.has_chest_data = has_chest_data
    self.has_wrist_data = wrist_rate is not None
    if not self.has_wrist_data and not self.has_chest_data:
        raise ValueError('You must set a chest sample rate, wrist sample rate, or both')
    if self.has_wrist_data:
      self.wrist_rate = wrist_rate if wrist_upscale is None else wrist_rate * wrist_upscale
      self.wrist_upscale = wrist_upscale
      self.wrist_baseline_info = None
      self.wrist_amusement = None
      self.wrist_stress = None
    if self.has_chest_data:
      self.chest_baseline_info = None
      self.chest_amusement = None
      self.chest_stress = None

  def process_raw(self, raw, for_wrist, baseline_times, amusement_times, stress_times):
    self._assert_has_data(for_wrist)
    if raw.shape[1] == 1:
      raw = raw.reshape(-1)
    if for_wrist and self.wrist_upscale is not None:
      raw = np.interp(np.arange(0, len(raw), 1/self.wrist_upscale), range(len(raw)), raw)

    def get_signals(rate):
      def segment(times):
        return raw[times[0]*rate:times[1]*rate]
      baseline = segment(baseline_times)
      amusement = segment(amusement_times)
      stress = segment(stress_times)
      return self.baseline_info_function(baseline, rate), self.process_function(amusement, rate), self.process_function(stress, rate)

    if for_wrist:
      self.wrist_baseline_info, self.wrist_amusement, self.wrist_stress = get_signals(self.wrist_rate)
    else:
      self.chest_baseline_info, self.chest_amusement, self.chest_stress = get_signals(700)

  def get_features(self, time, for_amusement):
    def get(for_wrist):
      sample_rate = self.wrist_rate if for_wrist else 700
      baseline_info = self.wrist_baseline_info if for_wrist else self.chest_baseline_info
      if for_wrist:
        signals = self.wrist_amusement if for_amusement else self.wrist_stress
      else:
        signals = self.chest_amusement if for_amusement else self.chest_stress

      end = sample_rate * time
      start = sample_rate * (time - self.window_size)
      if type(signals) is not pd.DataFrame:
        window = signals[start:end]
      else:
        window = signals.iloc[start:end]
        window = window.reset_index(drop=True)
      return self.feature_getter(window, baseline_info, sample_rate)

    features = {}
    if self.has_wrist_data:
      features.update(get(True))
    if self.has_chest_data:
      features.update(get(False))
    return features

  def _assert_has_data(self, for_wrist):
    if for_wrist and not self.has_wrist_data:
      raise ValueError("This measurement doesn't support wrist data")
    if not for_wrist and not self.has_chest_data:
      raise ValueError("This measurement doesn't support chest data")
