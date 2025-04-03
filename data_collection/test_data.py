import pandas as pd


df = pd.read_csv('pubmed_medical_reports.csv')


print(df.iloc[199]['abstract'])




