import pandas as pd
import pathlib

df = pd.read_csv('data/HumanActivityRecognition/metadata/sample_metadata.csv')
users = set()
envs = set()
devices = set()

for path in df['file_path']:
    parts = pathlib.Path(path).parts
    for part in parts:
        if part.startswith('user_'):
            users.add(part)
        elif part.startswith('env_'):
            envs.add(part)
        elif part.startswith('device_'):
            devices.add(part)

print("Users:", users)
print("Envs:", envs)
print("Devices:", devices)
