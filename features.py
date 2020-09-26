import numpy as np
import pandas as pd

# make sure these are the same as in the pickle_to_csv.py
EDA = 'EDA'
ACC = 'ACC'
TEMP = 'TEMP'
ECG = 'ECG'
EMG = 'EMG'
RESP = 'Resp'
BVP = 'BVP'

def get_signal_features(chest_signals, wrist_signals):
  result = {name: np.max(signal) for name, signal in chest_signals.items()}
  result.update({name: np.max(signal) for name, signal in wrist_signals.items()})
  return result
  # FIXME calculate real features for all signals
