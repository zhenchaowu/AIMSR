import itertools
import numpy as np
import pandas as pd




class AugmentedDataset:
    def __init__(self, sessions):
        self.sessions = sessions


    def __getitem__(self, idx):
        seq = self.sessions[0][idx]
        label = self.sessions[1][idx]
        return seq, label

    def __len__(self):
        return len(self.sessions[0])
