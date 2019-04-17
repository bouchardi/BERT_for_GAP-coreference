'''
Use this script to convert kaggle submission to file needed for gap_scorer.py

Usage of gap_scorer.py:
    python3 gap_scorer.py --gold_tsv gap-test.tsv --system_tsv system.tsv

'''

import pandas as pd

# read in kaggle submission
df = pd.read_csv('../kaggle_gap-test.tsv', delimiter='\t', encoding='utf-8')

# convert df to nparray
df_values = df[['A','B','NEITHER']].values

# if the MAX of the values equals the compared value then TRUE, else FALSE
nparray = df_values.max(1,keepdims=True) == df_values

# make new dataframe with new values
columns=['A-coref','B-coref', 'NEITHER']

# new dataframe
df_ = pd.DataFrame(data=nparray, columns=columns)

# insert columns with IDs
df_.insert(loc=0, column='ID', value=df['ID'])

# save to file
df_.to_csv("system.tsv", sep='\t', encoding='utf-8', index=False)

