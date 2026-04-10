import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class EnhancedTimeSeriesDataset(Dataset):
    def __init__(self, df, sequence_length=8, prediction_length=1,
                 augment_data=True, focus_classes=None,
                 scaler=None, label_encoder=None):
        self.df = df.copy()
        self.labels = self.df.iloc[:, 0].values
        self.time_series = self.df.iloc[:, 1:].values
        self.augment_data = augment_data
        self.focus_classes = focus_classes

        if label_encoder is not None:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(self.labels)
        else:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        self.num_classes = len(self.label_encoder.classes_)

        class_counts = np.bincount(self.encoded_labels)


        total_samples = len(self.encoded_labels)
        self.class_weights = total_samples / (self.num_classes * class_counts)
        self.class_weights = self.class_weights / self.class_weights.sum()

        if self.focus_classes is not None:
            focus_indices = [np.where(self.label_encoder.classes_ == str(cls))[0][0]
                             for cls in focus_classes if str(cls) in self.label_encoder.classes_]
            for idx in focus_indices:
                self.class_weights[idx] *= 5.0
            self.class_weights = self.class_weights / self.class_weights.sum()

        if scaler is not None:
            self.scaler = scaler
            self.scaled_data = self.scaler.transform(self.time_series)
        else:
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.time_series)

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.feature_dim = 1

        print(f"时间序列长度: {self.time_series.shape[1]}")
        print(f"输入序列长度: {sequence_length}")
        print(f"预测长度: {prediction_length}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_seq = self.scaled_data[idx, :self.sequence_length]
        target_seq = self.scaled_data[idx, self.sequence_length:self.sequence_length + self.prediction_length]
        class_label = self.encoded_labels[idx]

        if self.augment_data and torch.is_tensor(idx):
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.01, input_seq.shape)
                input_seq = input_seq + noise
            if np.random.random() > 0.5:
                scale_factor = np.random.uniform(0.95, 1.05)
                input_seq = input_seq * scale_factor

        input_seq = input_seq.reshape(-1, self.feature_dim)

        return (
            torch.FloatTensor(input_seq),
            torch.FloatTensor(target_seq),
            torch.LongTensor([class_label])
        )

    def get_scaler(self):
        return self.scaler

    def get_label_encoder(self):
        return self.label_encoder