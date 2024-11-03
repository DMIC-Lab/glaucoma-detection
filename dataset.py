from torch.utils.data import Dataset,DataLoader,random_split
import os
from PIL import Image
import numpy as np
import psutil
from scipy.ndimage import zoom
import torch



class GlaucomaDataset(Dataset):
    def __init__(self, path, df, additionalFeatures = {},transform=None, mode='binary', limit=0):
        self.path = path
        self.df = df
        self.d = additionalFeatures
        if limit:
            self.imagePaths = [os.path.join(path, file) for file in os.listdir(path)][:limit]
        else:
            self.imagePaths = [os.path.join(path, file) for file in os.listdir(path)]

        if mode != 'binary':
            # Create a set of Eye IDs with 'Final Label' == 'RG'
            valid_eye_ids = set(df.loc[df['Final Label'] == 'RG', 'Eye ID'])
            
            # Filter self.imagePaths based on whether the basename is in the valid_eye_ids
            self.imagePaths = [val for val in self.imagePaths if os.path.basename(val).split('.')[0] in valid_eye_ids] 
        self.transform = transform
        self.mode = mode

        # Initialize an empty cache (lazy loading)
        self.cache = {}

        # Define the columns order for multi-class labels
        self.multi_class_columns = [
            'Final ANRS', 'Final ANRI', 'Final RNFLDS', 'Final RNFLDI', 
            'Final BCLVS', 'Final BCLVI', 'Final NVT', 'Final DH', 
            'Final LD', 'Final LC'
        ]

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        filename = os.path.basename(self.imagePaths[index]).split('.')[0]
        dfEntry = self.df[self.df['Eye ID'] == filename]
        image = np.load(self.imagePaths[index])

        # Process the image (transpose, convert to tensor) and apply transforms
        image = self.prepImage(image)

        if self.mode == 'binary':
            label = torch.tensor((dfEntry['Final Label'] == 'RG').to_numpy(dtype=np.float64))
            return image,label
        else:
            # Consistently load multi-class labels in the defined order
            label = torch.tensor(
                [dfEntry[col].values[0] for col in self.multi_class_columns],
                dtype=torch.int8
            )
            if filename in self.d:
                additionalFeatures = torch.tensor(self.d[filename])
            else:
                additionalFeatures = torch.tensor((0,0,0,0,0))
            return image,label,additionalFeatures


    def getWeights(self):
        # Calculate the sum of each column (count of positive instances)
        column_sums = self.df[self.multi_class_columns].sum()
        
        # Calculate the total number of instances for each column
        total_counts = len(self.df)
        
        # Compute inverse weights for each column in the specified order
        weights = [total_counts / column_sums[col] if column_sums[col] > 0 else 1.0 for col in self.multi_class_columns]
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        normalized_weights = [weight / weight_sum for weight in weights]
        
        # Return as a tensor for compatibility with BCE loss
        return torch.tensor(normalized_weights, dtype=torch.float)

    def prepImage(self, image):
        # Process the raw image (transpose, convert to tensor)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image.float()

