import torch
from torch import nn

class BrainModule(nn.Module):
    def __init__(self, input_channels=272, output_size=768): # output_size=91168 for vdvae
        super().__init__()
        # res_dilated params
        kernel_size = 3
        dilation1 = 4
        dilation2 = 8
        dilation3 = 2
        dilation4 = 16
        dilation5 = 32
        dilation6 = 2
        padding1 = (kernel_size - 1) * dilation1 // 2
        padding2 = (kernel_size - 1) * dilation2 // 2
        padding3 = (kernel_size - 1) * dilation3 // 2
        padding4 = (kernel_size - 1) * dilation4 // 2
        padding5 = (kernel_size - 1) * dilation5 // 2
        padding6 = (kernel_size - 1) * dilation6 // 2

        self.spatial_attention = nn.Conv1d(input_channels, 270, kernel_size=1, stride=1) # original kernel_size=3, stride=1, padding=1
        self.linear_proj1 = nn.Conv1d(270, 270, kernel_size=1, stride=1) # original kernel_size=3, stride=1, padding=1
        self.linear_proj2 = nn.Conv1d(270, 270, kernel_size=1, stride=1) # original kernel_size=3, stride=1, padding=1
        # self.res_dilated_conv1 = nn.Conv1d(270, 320, kernel_size=3, stride=1, padding=1)
        self.res_dilated_conv1 = nn.ModuleList([nn.Sequential(nn.ConstantPad1d(padding1, 0), nn.Conv1d(270, 320, kernel_size=3, dilation=dilation1, stride=1, padding=0), nn.BatchNorm1d(320), nn.GELU()),
                                                nn.Sequential(nn.ConstantPad1d(padding2, 0), nn.Conv1d(320, 320, kernel_size=3, dilation=dilation2, stride=1, padding=0), nn.BatchNorm1d(320), nn.GELU()),
                                                nn.Sequential(nn.ConstantPad1d(padding3, 0), nn.Conv1d(320, 640, kernel_size=3, dilation=dilation3, stride=1, padding=0), nn.BatchNorm1d(640), nn.GLU(dim=1)),])
        # self.res_dilated_conv2 = nn.Conv1d(320, 320, kernel_size=3, stride=1, padding=1)
        
        self.res_dilated_conv2 = nn.ModuleList([nn.Sequential(nn.ConstantPad1d(padding4, 0), nn.Conv1d(320, 320, kernel_size=3, dilation=dilation4, stride=1, padding=0), nn.BatchNorm1d(320), nn.GELU()),
                                                nn.Sequential(nn.ConstantPad1d(padding5, 0), nn.Conv1d(320, 320, kernel_size=3, dilation=dilation5, stride=1, padding=0), nn.BatchNorm1d(320), nn.GELU()),
                                                nn.Sequential(nn.ConstantPad1d(padding6, 0), nn.Conv1d(320, 640, kernel_size=3, dilation=dilation6, stride=1, padding=0), nn.BatchNorm1d(640), nn.GLU(dim=1)),])
        self.linear_proj3 = nn.Conv1d(320, 2048, kernel_size=1, stride=1) # original kernel_size=3, stride=1, padding=1
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.temporal_aggregation = nn.AdaptiveAvgPool1d(1)
        # self.mlp_proj = nn.Linear(2048, output_size)
        self.mlp_proj = nn.Sequential(nn.LayerNorm(2048), nn.GELU(), nn.Linear(2048,output_size))

    def forward(self, x):
        x = self.spatial_attention(x)
        x = self.gelu(x)
        x = self.linear_proj1(x)
        # x = self.relu(x)
        x = self.linear_proj2(x)
        # x = self.relu(x)
        # x = self.res_dilated_conv1(x)
        # x = self.gelu(x)
        for i_layer, layer in enumerate(self.res_dilated_conv1):
            x = layer(x)
            if i_layer != 0:
                x += residual
            residual = x
        # x = self.res_dilated_conv2(x)
        # x = self.gelu(x)
        residual = x
        for i_layer, layer in enumerate(self.res_dilated_conv2):
            x = layer(x)
            x += residual
            residual = x
        x = self.linear_proj3(x)
        x = self.gelu(x)
        x = self.temporal_aggregation(x) # reducing the time length to 1
        # x = x.view(x.size(0), -1)  # Flatten out the time dimension
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.mlp_proj(x)
        return x

# class BrainModule(nn.Module):
#     def __init__(self, input_channels=272, output_size=768): # output_size=91168 for vdvae
#         super().__init__()
#         self.spatial_attention = nn.Conv1d(input_channels, 270, kernel_size=3, stride=1, padding=1) # original kernel_size=3, stride=1, padding=1
#         self.linear_proj1 = nn.Conv1d(270, 270, kernel_size=3, stride=1, padding=1) # original kernel_size=3, stride=1, padding=1
#         self.linear_proj2 = nn.Conv1d(270, 270, kernel_size=3, stride=1, padding=1) # original kernel_size=3, stride=1, padding=1
#         self.res_dilated_conv1 = nn.Conv1d(270, 320, kernel_size=3, stride=1, padding=1)
#         self.res_dilated_conv2 = nn.Conv1d(320, 320, kernel_size=3, stride=1, padding=1)
#         self.linear_proj3 = nn.Conv1d(320, 2048, kernel_size=3, stride=1, padding=1) # original kernel_size=3, stride=1, padding=1
#         self.relu = nn.ReLU()
#         self.temporal_aggregation = nn.AdaptiveAvgPool1d(1)
#         self.mlp_proj = nn.Linear(2048, output_size)
#         # self.mlp_proj = nn.Sequential(nn.LayerNorm(2048), nn.GELU(), nn.Linear(2048,output_size))

#     def forward(self, x):
#         x = self.spatial_attention(x)
#         x = self.relu(x)
#         x = self.linear_proj1(x)
#         x = self.relu(x)
#         x = self.linear_proj2(x)
#         x = self.relu(x)
#         x = self.res_dilated_conv1(x)
#         x = self.relu(x)
#         x = self.res_dilated_conv2(x)
#         x = self.relu(x)
#         x = self.linear_proj3(x)
#         x = self.temporal_aggregation(x) # reducing the time length to 1
#         # x = x.view(x.size(0), -1)  # Flatten out the time dimension
#         x = x.squeeze(-1)  # Remove the last dimension
#         x = self.mlp_proj(x)
#         return x

# class BrainModule(nn.Module): 
#     def __init__(self, input_channels=272, hidden_size=270, output_size=768):
#         super().__init__()
#         self.conv1 = nn.Conv1d(input_channels, 270, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(270, 270, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(270, 320, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv1d(320, 320, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv1d(320, 2048, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(2048, output_size)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = self.conv5(x)
#         x = self.avg_pool(x) # reducing the time length to 1
#         # x = x.view(x.size(0), -1)  # Flatten out the time dimension
#         x = x.squeeze(-1)  # Remove the last dimension
#         x = self.fc(x)
#         return x