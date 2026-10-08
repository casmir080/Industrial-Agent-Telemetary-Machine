import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CMAPSSDataset(Dataset):
    """PyTorch Dataset wrapper for sequence windows."""
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class CMAPSSDataPipeline:
    """End-to-End data ingestion and windowing pipeline for NASA FD001."""
    def __init__(self, data_dir: str = "data/raw", sequence_length: int = 30):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # Standard CMAPSS column mapping
        self.columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                       [f'sensor_{i}' for i in range(1, 22)]
        
        # Exclude metadata from input features
        self.feature_cols = self.columns[2:] 

    def _check_data_exists(self):
        files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
        for f in files:
            if not (self.data_dir / f).exists():
                raise FileNotFoundError(
                    f"CRITICAL: Dataset file missing -> {f}. "
                    "Download the NASA CMAPSS dataset from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps "
                    "and place the extracted text files in data/raw/"
                )

    def _load_data(self, filename: str) -> pd.DataFrame:
        file_path = self.data_dir / filename
        return pd.read_csv(file_path, sep=r'\s+', header=None, names=self.columns)

    def _generate_train_sequences(self, df: pd.DataFrame):
        sequences, targets = [], []
        for unit_id in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit_id]
            features = unit_data[self.feature_cols].values
            rul = unit_data['RUL'].values
            
            # Sliding window construction
            for i in range(len(unit_data) - self.sequence_length + 1):
                sequences.append(features[i : i + self.sequence_length])
                targets.append(rul[i + self.sequence_length - 1])
                
        return np.array(sequences), np.array(targets)

    def _generate_test_sequences(self, df: pd.DataFrame, test_rul: np.ndarray):
        sequences, targets = [], []
        for unit_id in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit_id]
            features = unit_data[self.feature_cols].values
            
            # CMAPSS test set requires evaluating only the last sequence of the unit
            if len(unit_data) >= self.sequence_length:
                sequences.append(features[-self.sequence_length:])
            else:
                # Zero-pad if the engine history is shorter than the sequence length window
                pad_len = self.sequence_length - len(unit_data)
                padded = np.pad(features, ((pad_len, 0), (0, 0)), 'constant', constant_values=0)
                sequences.append(padded)
                
            targets.append(test_rul[unit_id - 1])
                
        return np.array(sequences), np.array(targets)

    def prepare_data(self, batch_size: int = 64):
        self._check_data_exists()
        
        train_df = self._load_data("train_FD001.txt")
        test_df = self._load_data("test_FD001.txt")
        rul_df = pd.read_csv(self.data_dir / "RUL_FD001.txt", sep=r'\s+', header=None, names=['RUL'])
        
        # Calculate continuous RUL targets for training: max_cycle - current_cycle
        rul_max = pd.DataFrame(train_df.groupby('unit_number')['time_in_cycles'].max()).reset_index()
        rul_max.columns = ['unit_number', 'max']
        train_df = train_df.merge(rul_max, on=['unit_number'], how='left')
        train_df['RUL'] = train_df['max'] - train_df['time_in_cycles']
        train_df.drop('max', axis=1, inplace=True)
        
        # Fit scaler strictly on training distribution to prevent data leakage
        train_df[self.feature_cols] = self.scaler.fit_transform(train_df[self.feature_cols])
        test_df[self.feature_cols] = self.scaler.transform(test_df[self.feature_cols])
        
        # Build sequences
        train_seq, train_targets = self._generate_train_sequences(train_df)
        test_seq, test_targets = self._generate_test_sequences(test_df, rul_df['RUL'].values)
        
        # Wrap in PyTorch DataLoaders
        train_dataset = CMAPSSDataset(train_seq, train_targets)
        test_dataset = CMAPSSDataset(test_seq, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader

if __name__ == "__main__":
    print("Initializing CMAPSS Data Pipeline Verification...")
    try:
        pipeline = CMAPSSDataPipeline(sequence_length=30)
        train_loader, test_loader = pipeline.prepare_data(batch_size=64)
        
        X_batch, y_batch = next(iter(train_loader))
        
        print(f"[SUCCESS] Pipeline executed cleanly.")
        print(f"[METRICS] Batches loaded. X tensor shape: {X_batch.shape} | Y tensor shape: {y_batch.shape}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")