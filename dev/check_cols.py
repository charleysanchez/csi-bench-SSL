import pandas as pd
import glob

for csv_path in glob.glob('data/*/metadata/sample_metadata.csv'):
    df = pd.read_csv(csv_path, nrows=1)
    print(csv_path)
    print("Has user:", 'user' in df.columns)
    print("Has env:", 'environment' in df.columns)
    print("Has device:", 'device' in df.columns)
    print("---")
