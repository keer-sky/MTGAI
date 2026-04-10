import numpy as np
import torch
import pandas as pd

def create_sample_data():
    np.random.seed(42)
    n_samples = 200
    time_steps = 60

    data = []
    labels = []

    for i in range(n_samples):
        if i % 4 == 0:
            t = np.linspace(0, 4 * np.pi, time_steps)
            series = np.sin(t) + np.random.normal(0, 0.1, time_steps)
            label = '-1'
        elif i % 4 == 1:
            t = np.linspace(0, 4 * np.pi, time_steps)
            series = np.cos(t) + np.random.normal(0, 0.1, time_steps)
            label = '0'
        elif i % 4 == 2:
            series = np.linspace(0, 1, time_steps) + np.random.normal(0, 0.1, time_steps)
            label = 'Class_A'
        else:
            series = np.linspace(-1, 1, time_steps) + np.random.normal(0, 0.1, time_steps)
            label = 'Class_B'

        data.append(series)
        labels.append(label)

    df = pd.DataFrame(data)
    df.insert(0, 'Label', labels)
    df.to_excel('sample_data.xlsx', index=False)
    return 'sample_data.xlsx'


def load_model_from_checkpoint(checkpoint_path, model_class, device, **model_kwargs):
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model