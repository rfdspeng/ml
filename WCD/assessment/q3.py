# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:20:55 2024

@author: Ryan Tsai
"""

import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    df = pd.read_csv(data_url)
    
    females_idx = np.nonzero(df.get('Sex') == 'female')
    print(females_idx)
    
    df_females = df.get('Sex').iloc[females_idx]
    print(df_females)