# https://predictioncenter.org/casp15/multimer_results.cgi?target=H1106&view=txt
import requests
import os
import pandas as pd


d = "/root/EnQA/casp15_data"
case_ids = [i for i in os.listdir(d) if os.path.isdir(os.path.join(d, i))]
print(case_ids)
for case in case_ids:
    url = f"https://predictioncenter.org/casp15/multimer_results.cgi?target={case}&view=txt"
    r = requests.get(url)
    with open(f"{case}.txt", "w") as f:
        f.write(r.text)


def read_table(path):
    df = pd.DataFrame(columns=['model', 'lDDToligo'])
    with open(path, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        cols = line.split()
        # if line[0] is a string digit
        if not (line[0]).isdigit():
            continue
        if len(cols) < 10:
            continue
        model_name = cols[1]
        try:
            lDDToligo = float(cols[16])
        except ValueError:
            continue
        df = df.append({'model': model_name, 'lDDToligo': lDDToligo}, ignore_index=True)
    return df


def join_df(true_df, pred_df):
    shared_df = true_df.merge(pred_df, on='model', how='inner', suffixes=('_True', '_EnQA-MSA'))
    return shared_df