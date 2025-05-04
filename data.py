import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class StockDataset(Dataset):
    """Dataset for loading stock chart images and tabular data with true temporal sequences."""
    
    def __init__(
        self,
        csv_file: str,
        base_dir: str = "./",
        image_transform=None,
        max_seq_len: int = 14,
        standardize_targets: bool = False,
    ):
        super().__init__()
        """
        Args:
            csv_file: Path to the CSV file with dataset info.
            base_dir: Base directory for the dataset.
            image_transform: Optional transform to be applied on the images.
            max_seq_len: Maximum sequence length for tabular data.
        """
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.image_transform = image_transform
        self.max_seq_len = max_seq_len
        self.standardize_targets = standardize_targets

        
        # Independent variables
        self.tabular_cols = [col for col in self.data.columns 
                           if col.startswith(('MACRO_', 'SMA_', 'RSI', 'EMA_', 'MACD', 'BB_', 
                                              'Volatility', 'Open', 'High', 'Close', 'Volume', "Lagged_"))]

        # Dependent variables
        self.target_cols = [col for col in self.data.columns 
                          if col.startswith('TARGET_Return_')]

        self.target_stats = {}
        for target_col in ['TARGET_Return_1d', 'TARGET_Return_5d', 'TARGET_Return_10d']:
            if target_col in self.data.columns:
                mean = self.data[target_col].mean()
                std = self.data[target_col].std()
                self.target_stats[target_col] = {'mean': mean, 'std': std}
                print(f"Target {target_col}: mean={mean:.4f}, std={std:.4f}")
        
        # Create a mapping of window_ids to their full temporal sequences
        self.temporal_sequences = {}
        
        # Initialize drop_indices before calling _load_temporal_sequences
        self.drop_indices = []
        
        # Now call the method that uses drop_indices
        self._load_temporal_sequences()
        
        print(f"Loaded dataset with {len(self.data)} samples")
        print(f"Found {len(self.tabular_cols)} tabular features")
        print(f"Found {len(self.target_cols)} target variables")
    
    def _load_temporal_sequences(self):
        """
        Load temporal sequences with synthetic data handling
        """
        for idx, row in tqdm(self.data.iterrows(), desc="Loading temporal sequences", total=len(self.data)):
            window_id = row['window_id'] if 'window_id' in row else str(idx)
            
            # Try to find the aligned data file for this window
            aligned_file = os.path.join(self.base_dir, "integrated_stock_macro_data/aligned_data", f"{window_id}_aligned.csv")
            
            if os.path.exists(aligned_file):
                try:
                    aligned_data = pd.read_csv(aligned_file, index_col=0)
                    
                    # Extract the features for the last max_seq_len days
                    seq_data = aligned_data[self.tabular_cols].iloc[-self.max_seq_len:].values
                    
                    # Handle case where sequence is shorter than max_seq_len
                    if len(seq_data) < self.max_seq_len:
                        # Pad with zeros at the beginning
                        padded_seq = np.zeros((self.max_seq_len, len(self.tabular_cols)))
                        padded_seq[-len(seq_data):] = seq_data
                        seq_data = padded_seq
                    
                    # Mark as real data
                    self.temporal_sequences[window_id] = {
                        'data': seq_data,
                        'is_synthetic': False
                    }
                except Exception as e:
                    print(f"Error loading aligned data for {window_id}: {e}")
                    # Drop this sample instead of using synthetic data
                    self.drop_indices.append(idx)
            else:
                # Instead of creating synthetic data, mark for dropping
                self.drop_indices.append(idx)
        
        # After processing, remove all dropped indices from self.data
        if self.drop_indices:
            print(f"Dropping {len(self.drop_indices)} samples with missing aligned data")
            self.data = self.data.drop(self.drop_indices).reset_index(drop=True)
    
    def _create_fallback_sequence(self, window_id, row):
        """Create a fallback sequence when aligned data is not available."""
        print(f"Aligned data not found for {window_id}, using synthetic sequence")
        
        # Try to extract tabular features from the current row
        try:
            # Convert all values to numeric, replacing non-numeric with 0.0
            tabular_values = []
            for col in self.tabular_cols:
                try:
                    value = float(row[col])
                    tabular_values.append(value)
                except (ValueError, TypeError):
                    tabular_values.append(0.0)
            
            # Create a synthetic sequence by adding some random walk to the current values
            base_values = np.array(tabular_values)
            sequence = np.zeros((self.max_seq_len, len(self.tabular_cols)))
            
            # Generate a random walk sequence
            for i in range(self.max_seq_len):
                # Add decreasing random noise as we approach the end of the sequence
                # This simulates how older data points are less relevant to the final prediction
                noise_scale = 0.05 * (self.max_seq_len - i) / self.max_seq_len
                random_noise = np.random.randn(len(self.tabular_cols)) * noise_scale
                sequence[i] = base_values * (1.0 - noise_scale) + random_noise
            
            self.temporal_sequences[window_id] = sequence
        except Exception as e:
            print(f"Error creating synthetic sequence for {window_id}: {e}")
            # Last resort: just create a zero sequence
            self.temporal_sequences[window_id] = np.zeros((self.max_seq_len, len(self.tabular_cols)))
    
    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.data.iloc[idx]
        window_id = row['window_id'] if 'window_id' in row else str(idx)
        
        # Load image
        img_path = os.path.join(self.base_dir, row['chart_path'])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            image = torch.zeros(3, 224, 224)
        
        # Load temporal sequence
        try:
            seq_info = self.temporal_sequences.get(window_id)
            if seq_info is None:
                print(f"Warning: No sequence found for {window_id}")
                seq_data = np.zeros((self.max_seq_len, len(self.tabular_cols)))
            else:
                seq_data = seq_info['data']
            
            # Convert to tensor
            seq_features = torch.tensor(seq_data, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading sequence for {window_id}: {e}")
            seq_features = torch.zeros((self.max_seq_len, len(self.tabular_cols)), dtype=torch.float32)
        
        target_values = {}
        for target_col, key in zip(['TARGET_Return_1d', 'TARGET_Return_5d', 'TARGET_Return_10d'], 
                                 ['return_1d', 'return_5d', 'return_10d']):
            try:
                value = float(row[target_col])
                # Store the raw return value (we'll convert to direction in the loss function)
                target_values[key] = torch.tensor(value, dtype=torch.float32)
            except (ValueError, TypeError):
                target_values[key] = torch.tensor(0.0, dtype=torch.float32)
        
        return {
            'image': image,
            'tabular': seq_features,  
            'targets': target_values,
            'window_id': window_id
        }

    @classmethod
    def get_transforms(cls, is_training: bool = True) -> transforms.Compose:
        """Get image transformations based on whether we're training or evaluating."""
        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                # No horizontal flips - they destroy temporal semantics
                # Add safe affine transforms instead
                transforms.RandomAffine(
                    degrees=(-2, 2),  # Slight rotation
                    translate=(0.02, 0.02),  # Small translation
                    scale=(0.98, 1.02),  # Slight scaling
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    @classmethod
    def collate_fn(cls, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching samples."""
        images = torch.stack([item['image'] for item in batch])
        tabular = torch.stack([item['tabular'] for item in batch])
        
        targets = {
            'return_1d': torch.tensor([item['targets']['return_1d'] for item in batch]),
            'return_5d': torch.tensor([item['targets']['return_5d'] for item in batch]),
            'return_10d': torch.tensor([item['targets']['return_10d'] for item in batch])
        }
        
        window_ids = [item['window_id'] for item in batch]
        
        return {
            'images': images,
            'tabular': tabular,
            'targets': targets,
            'window_ids': window_ids
        }
