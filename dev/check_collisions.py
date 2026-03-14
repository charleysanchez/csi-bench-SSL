import pandas as pd
import glob
import re

for csv_path in glob.glob('data/*/metadata/sample_metadata.csv'):
    df = pd.read_csv(csv_path)
    if 'user' in df.columns:
        users = df['user'].dropna().unique()
        suffixes = []
        for u in users:
            m = re.search(r'([0-9]+)$', str(u))
            if m:
                suffixes.append(int(m.group(1)))
        
        if len(suffixes) != len(set(suffixes)):
            print(f"COLLISION in {csv_path}: {users}")
    if 'environment' in df.columns:
        envs = df['environment'].dropna().unique()
        suffixes = []
        for e in envs:
            m = re.search(r'([0-9]+)$', str(e))
            if m:
                suffixes.append(int(m.group(1)))
        
        if len(suffixes) != len(set(suffixes)):
            print(f"COLLISION in {csv_path} (envs): {envs}")
