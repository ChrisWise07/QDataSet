import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F
import torch

TWO_PI = 2 * torch.pi


class SimpleANN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_noise_matrices: int,
        noise_matrix_dim: int,
    ):

        """
        Args:

            input_size (int): the length of the sequence.

            num_noise_matrices (int):
                the number of noise matrices to predict

            noise_matrix_dim (int):
                the dimension of the noise matrices to predict where
                matrices are assumed to be square

        """

        super().__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_noise_matrices * (noise_matrix_dim**2))

    def custom_activation(self, x: Tensor) -> Tensor:
        """
        Applies a custom activation function to the output of the
        neural network. The custom activation function is a sigmoid
        applied to all elements. Every element is multiplied by 2pi
        except for every third element which is between 0-1.

        Args:
            x (Tensor):
                a tensor of shape [
                    batch_size,
                    num_noise_matrices * noise_matrix_dim**2
                ]

        Returns:
            Tensor:
                a tensor of shape [
                    batch_size,
                    num_noise_matrices * noise_matrix_dim**2
                ]
        """
        x = TWO_PI * torch.sigmoid(x)
        x[:, 2::3] = x[:, 2::3] / TWO_PI
        return x

    def forward(
        self,
        time_series_squence: Tensor,
    ) -> Tensor:
        """
        Returns a tensor of shape:


        Args:
            time_series_squence (Tensor):
                a tensor of shape [batch_size, seq_len]

        Returns:
            Tensor:
                a tensor of shape: [
                    batch_size,
                    num_noise_matrices * (noise_matrix_dim**2)
                ]
        """
        x1 = F.relu(self.fc1(time_series_squence))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        return self.custom_activation(self.fc5(x4))
