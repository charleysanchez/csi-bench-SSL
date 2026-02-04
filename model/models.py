import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    """Multi-layer Perceptron for WiFi sensing"""
    def __init__(self, win_len=500, feature_size=232, num_classes=2):
        super(MLPClassifier, self).__init__()
        # Calculate input size but limit it to prevent memory issues
        input_size = min(win_len * feature_size, 10000)
        
        self.win_len = win_len
        self.feature_size = feature_size
        self.num_classes = num_classes
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def get_init_params(self):
        """Return the initialization parameters to support model cloning for few-shot learning"""
        return {
            'win_len': self.win_len,
            'feature_size': self.feature_size,
            'num_classes': self.num_classes
        }
        
    def forward(self, x):
        # Flatten input: [batch, channels, win_len, feature_size] -> [batch, win_len*feature_size]
        x = x.view(x.size(0), -1)
        # Limit input size if needed
        if x.size(1) > 10000:
            x = x[:, :10000]
        return self.fc(x)
    
