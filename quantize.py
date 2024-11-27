import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import argparse
from fairseq.models.wav2vec import utils as wav2vec_utils



def pad_to_multiple_custom(x, multiple, dim=-1, value=0):
    import math  # Ensure the math module is imported
    
    tsz = x.shape[dim]  # Get the size of the dimension
    if isinstance(tsz, Tensor):  # Ensure it's converted to an integer
        tsz = tsz.item()
    
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if remainder == 0:
        return x, 0

    pad = [0] * ((-1 - dim) * 2)  # Handle padding correctly for multiple dimensions
    pad[-2] = remainder
    x = F.pad(x, pad, value=value)
    return x, remainder


wav2vec_utils.pad_to_multiple = pad_to_multiple_custom


class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        task_arg = argparse.Namespace(task='audio_pretraining')
        task = fairseq.tasks.setup_task(task_arg)
        cp_path = '../ml/weights/xlsr2_300m.pt'   # Path to pre-trained model 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path], task=task)
        self.model = model[0].to(device)  # Move the model to the specified device only once
        self.model.encoder.required_seq_len_multiple = 1  # Set to 1 to disable padding
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # Ensure input is on the correct device
        input_data = input_data.to(self.device)

        # Adjust input shape to (batch, length) if necessary
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
                
        # Extract features [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


class PSFAN_Backend(nn.Module):
    def __init__(self, input_channels=128, num_classes=2):
        super(PSFAN_Backend, self).__init__()
        
        # First convolutional block with dilation rate = 1
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=3, dilation=1, padding=1)
        self.conv1x1_1 = nn.Conv1d(128, 128, kernel_size=1)
        self.conv3x3_1 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1x1_2 = nn.Conv1d(128, 128, kernel_size=1)
        self.attention1 = nn.Sigmoid()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Second convolutional block with dilation rate = 2
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, dilation=2, padding=2)
        self.conv1x1_3 = nn.Conv1d(128, 128, kernel_size=1)
        self.conv3x3_2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1x1_4 = nn.Conv1d(128, 128, kernel_size=1)
        self.attention2 = nn.Sigmoid()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Third convolutional block with dilation rate = 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, dilation=3, padding=3)
        self.conv1x1_5 = nn.Conv1d(256, 256, kernel_size=1)
        self.conv3x3_3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv1x1_6 = nn.Conv1d(256, 256, kernel_size=1)
        self.attention3 = nn.Sigmoid()
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Fourth convolutional block with dilation rate = 4
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, dilation=4, padding=4)
        self.conv1x1_7 = nn.Conv1d(256, 256, kernel_size=1)
        self.conv3x3_4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv1x1_8 = nn.Conv1d(256, 256, kernel_size=1)
        self.attention4 = nn.Sigmoid()
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)# Global Average Pooling layer for each block output
        self.gap1 = nn.AdaptiveAvgPool1d(1)
        self.gap2 = nn.AdaptiveAvgPool1d(1)
        self.gap3 = nn.AdaptiveAvgPool1d(1)
        self.gap4 = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc_concat = nn.Linear(128 + 128 + 256 + 256, 16)  # Concatenated GAP output to dense layer
        self.fc_out = nn.Linear(16, num_classes)  # Final output layer
        
        self.activation = nn.LeakyReLU(0.02)

    def forward(self, x):
        # First convolutional block with attention and pooling
        x1 = self.conv1(x)
        x1_attention = self.attention1(self.conv1x1_1(self.conv3x3_1(self.conv1x1_2(x1))))
        x1 = x1_attention * x1
        x1 = self.pool1(x1)
        x1_gap = self.gap1(x1).squeeze(-1)  # Apply GAP and remove last dimension to (batch, channels)

        # Second convolutional block with attention and pooling
        x2 = self.conv2(x1)
        x2_attention = self.attention2(self.conv1x1_3(self.conv3x3_2(self.conv1x1_4(x2))))
        x2 = x2_attention * x2
        x2 = self.pool2(x2)
        x2_gap = self.gap2(x2).squeeze(-1)

        # Third convolutional block with attention and pooling
        x3 = self.conv3(x2)
        x3_attention = self.attention3(self.conv1x1_5(self.conv3x3_3(self.conv1x1_6(x3))))
        x3 = x3_attention * x3
        x3 = self.pool3(x3)
        x3_gap = self.gap3(x3).squeeze(-1)

        # Fourth convolutional block with attention and pooling
        x4 = self.conv4(x3)
        x4_attention = self.attention4(self.conv1x1_7(self.conv3x3_4(self.conv1x1_8(x4))))
        x4 = x4_attention * x4
        x4 = self.pool4(x4)
        x4_gap = self.gap4(x4).squeeze(-1)

        # Concatenate the GAP outputs
        x_concat = torch.cat([x1_gap, x2_gap, x3_gap, x4_gap], dim=1)  # Shape: (batch, 768)

        # Fully connected layers for classification
        x = self.activation(self.fc_concat(x_concat))  # Dense layer with 16 units
        output = self.fc_out(x)  # Output layer with 2 units

        return output


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device
        
        # wav2vec 2.0 front-end
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128).to(device)  # Reduces dimensionality to 128 for compatibility

        # PSFAN backend
        self.backend = PSFAN_Backend(input_channels=128, num_classes=2).to(device)

    def forward(self, x):
        x = x.to(self.device)
        x_ssl_feat = self.ssl_model.extract_feat(x)
        x = self.LL(x_ssl_feat)
        x = x.transpose(1, 2)
        output = self.backend(x)
        return output

device = 'cpu'
model = Model(device)

dummy_input = torch.randn(1, 1000, 1, device=device)

model.eval()  # Set the model to evaluation mode
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

torch.save(model, 'quantized_full.pth')