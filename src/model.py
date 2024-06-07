import torch.nn as nn
import torch
from torchinfo import summary

from replay_buffer import ReplayBuffer


class CNN(nn.Module):
    def __init__(self, input_channels=4, action_dim=4):
        super(CNN, self).__init__()


        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1), 
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    
    
        

if __name__ == "__main__":

    lenet = CNN()

    summ =  summary(model=lenet, input_size=(1, 4, 84, 84), col_width=20,
                    col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], verbose=0)
    
    print(summ)
# ========================================================================================================================
# Layer (type (var_name))                  Input Shape          Output Shape         Param #              Trainable
# ========================================================================================================================
# CNN (CNN)                                [1, 4, 84, 84]       [1, 4]               --                   True
# ├─Sequential (feature_extraction)        [1, 4, 84, 84]       [1, 64, 7, 7]        --                   True
# │    └─Conv2d (0)                        [1, 4, 84, 84]       [1, 32, 20, 20]      8,224                True
# │    └─ReLU (1)                          [1, 32, 20, 20]      [1, 32, 20, 20]      --                   --
# │    └─Conv2d (2)                        [1, 32, 20, 20]      [1, 64, 9, 9]        32,832               True
# │    └─ReLU (3)                          [1, 64, 9, 9]        [1, 64, 9, 9]        --                   --
# │    └─Conv2d (4)                        [1, 64, 9, 9]        [1, 64, 7, 7]        36,928               True
# │    └─ReLU (5)                          [1, 64, 7, 7]        [1, 64, 7, 7]        --                   --
# ├─Sequential (fc)                        [1, 3136]            [1, 4]               --                   True
# │    └─Flatten (0)                       [1, 3136]            [1, 3136]            --                   --
# │    └─Linear (1)                        [1, 3136]            [1, 512]             1,606,144            True
# │    └─ReLU (2)                          [1, 512]             [1, 512]             --                   --
# │    └─Linear (3)                        [1, 512]             [1, 4]               2,052                True
# ========================================================================================================================
# Total params: 1,686,180
# Trainable params: 1,686,180
# Non-trainable params: 0
# Total mult-adds (M): 9.37
# ========================================================================================================================
# Input size (MB): 0.11
# Forward/backward pass size (MB): 0.17
# Params size (MB): 6.74
# Estimated Total Size (MB): 7.03
# ========================================================================================================================