import torch.nn as nn
from torch import nn, Tensor
import positional_encoder as pe
import torch.nn.functional as F
import torch
import numpy as np

TWO_PI = 2 * np.pi


def trace_distance(A, B):
    return torch.norm(A - B, p="fro")


def custom_loss(y_pred, y_true) -> Tensor:
    batch_size = y_pred.shape[0]
    loss = 0.0

    for i in range(batch_size):
        hat_V_X = y_pred[i, 0:4].reshape(2, 2)
        gt_V_X = y_true[i, 0:4].reshape(2, 2)
        hat_V_Y = y_pred[i, 4:8].reshape(2, 2)
        gt_V_Y = y_true[i, 4:8].reshape(2, 2)
        hat_V_Z = y_pred[i, 8:12].reshape(2, 2)
        gt_V_Z = y_true[i, 8:12].reshape(2, 2)

        loss += (
            trace_distance(hat_V_X, gt_V_X)
            + trace_distance(hat_V_Y, gt_V_Y)
            + trace_distance(hat_V_Z, gt_V_Z)
        )

    return loss / batch_size


def construct_V_O(psi, theta, Delta, mu, O_inv):
    Q = torch.zeros((2, 2), dtype=torch.float32)
    D = torch.zeros((2, 2), dtype=torch.float32)

    Q[0, 0] = torch.cos(theta) * torch.cos(Delta)
    Q[0, 1] = torch.cos(theta) * torch.sin(Delta)
    Q[1, 0] = -torch.sin(theta) * torch.sin(Delta)
    Q[1, 1] = torch.sin(theta) * torch.cos(Delta)

    Q = torch.mm(
        Q,
        torch.tensor(
            [[torch.cos(psi), 0], [0, torch.cos(psi)]], dtype=torch.float32
        ),
    )
    Q = torch.mm(
        torch.tensor(
            [[torch.cos(psi), 0], [0, torch.cos(psi)]], dtype=torch.float32
        ),
        Q,
    )

    D[0, 0] = mu
    D[1, 1] = -mu

    V_O = torch.mm(O_inv, torch.mm(Q, torch.mm(D, Q.t())))
    return V_O


for i in range(y_pred.shape[0]):
    psi_X = y_pred[i, 0]
    theta_X = y_pred[i, 1]
    Delta_X = y_pred[i, 2]
    mu_X = y_pred[i, 3]

    psi_Y = y_pred[i, 4]
    theta_Y = y_pred[i, 5]
    Delta_Y = y_pred[i, 6]
    mu_Y = y_pred[i, 7]

    psi_Z = y_pred[i, 8]
    theta_Z = y_pred[i, 9]
    Delta_Z = y_pred[i, 10]
    mu_Z = y_pred[i, 11]

    V_O_X = construct_V_O(psi_X, theta_X, Delta_X, mu_X, O_inv)
    V_O_Y = construct_V_O(psi_Y, theta_Y, Delta_Y, mu_Y, O_inv)
    V_O_Z = construct_V_O(psi_Z, theta_Z, Delta_Z, mu_Z, O_inv)


class EncoderWithMLP(nn.Module):

    """
    This class implements an encoder only transformer model that is
    used for analysis of times series data. The output of the encoder is
    then fed to a complex valued neural network. The final output of
    the complex valued neural network are the noise encoding matrices.

    The encoder code borrows from a transformer tutorial
    by Ludvigssen [1]. Hyperparameter and architecture details are
    borrowed from Vaswani et al (2017) [2].

    [1] Ludvigsen, K.G.A. (2022)
    'How to make a pytorch transformer for time series forecasting'.
    Medium. Towards Data Science.
    Available at: https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e (Accessed: February 10, 2023).

    [2] Vaswani, A. et al. (2017)
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint].
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: February 10, 2023).

    """

    def __init__(
        self,
        input_size: int,
        batch_first: bool,
        num_noise_matrices: int,
        noise_matrix_dim: int,
        d_model: int = 512,
        n_encoder_layers: int = 4,
        n_heads: int = 8,
        dropout_encoder: float = 0.2,
        dropout_pos_enc: float = 0.1,
        dim_feedforward_encoder: int = 2048,
    ):

        """
        Args:

            input_size (int): number of input variables.

            batch_first (bool):
                if True, the batch dimension is the first in the input
                and output tensors

            num_noise_matrices (int):
                the number of noise matrices to predict

            noise_matrix_dim (int):
                the dimension of the noise matrices to predict where
                matrices are assumed to be square

            d_model (int):
                All sub-layers in the model produce outputs
                of dimension d_model

            n_encoder_layers (int):
                number of stacked encoder layers in the encoder

            n_heads (int): the number of attention heads

            dropout_encoder (float): the dropout rate of the encoder

            dropout_pos_enc (float): t
                he dropout rate of the positional encoder

            dim_feedforward_encoder (int):
                number of neurons in the linear layer of the encoder
        """

        super().__init__()

        self.encoder_input_layer = nn.Linear(
            in_features=input_size, out_features=d_model
        )

        self.positional_encoding_layer = pe.PositionalEncoder(
            d_model=d_model, dropout=dropout_pos_enc
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None
        )

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_noise_matrices * noise_matrix_dim**2)

    def custom_activation(self, x):
        x[:, 3::4] = torch.sigmoid(x[:, 3::4])
        x[:, 0:3] = TWO_PI * torch.sigmoid(x[:, 0:3])
        x[:, 4:7] = TWO_PI * torch.sigmoid(x[:, 4:7])
        x[:, 8:11] = TWO_PI * torch.sigmoid(x[:, 8:11])
        return x

    def forward(
        self,
        time_series_squence: Tensor,
    ) -> Tensor:
        """
        Returns a tensor of shape:


        Args:
            time_series_squence (Tensor):
                a tensor of shape [batch_size, seq_len, 1]

        Returns:
            Tensor:
                a tensor of shape: [
                    batch_size,
                    num_noise_matrices * noise_matrix_dim**2
                ]
        """

        encoder_input = self.encoder_input_layer(time_series_squence)
        positional_encoding = self.positional_encoding_layer(encoder_input)
        encoder_output = self.encoder(positional_encoding)
        x1 = F.relu(self.fc1(encoder_output))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        return self.custom_activation(self.fc4(x3))


model = EncoderWithMLP(
    input_size=1024,
    batch_first=True,
    num_noise_matrices=3,
    noise_matrix_dim=2,
)

criterion = custom_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

nn.MSELoss()(y_pred, y_test)

for epoch in range(100):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


y_pred = model(X_test)
