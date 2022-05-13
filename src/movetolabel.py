import pandas as pd
import os

df = pd.read_csv('src\labels\AllLabels.csv')

for _, row in df.iterrows():
    f = row['ClipID']
    l = row['Engagement']
    try:
        os.replace(f'src\datasetimage\{f}', f'src\datasetimage\{l}\{f}')
    except FileNotFoundError:
        print("File ",str(f)," not found in folder.")