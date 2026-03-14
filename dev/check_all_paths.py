import pandas as pd
import pathlib
import glob

users = set()
envs = set()
devices = set()

for csv_path in glob.glob('data/*/metadata/sample_metadata.csv'):
    df = pd.read_csv(csv_path)
    if 'file_path' in df.columns:
        for path in df['file_path']:
            parts = pathlib.Path(path).parts
            for part in parts:
                if part.startswith('user_'):
                    users.add(part)
                elif part.startswith('env_'):
                    envs.add(part)
                elif part.startswith('device_'):
                    devices.add(part)

print("Users:", sorted(list(users)))
print("Envs:", sorted(list(envs)))
print("Devices:", sorted(list(devices)))
