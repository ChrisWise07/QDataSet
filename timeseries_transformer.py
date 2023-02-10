import torch.nn as nn
from torch import nn, Tensor
import positional_encoder as pe
import torch.nn.functional as F


class TimeSeriesToQuantumNoiseEncoders(nn.Module):

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

    def forward(
        self,
        time_series_squence: Tensor,
    ) -> Tensor:
        """
        Returns a tensor of shape:

        [
            batch_size,
            num_noise_matrices,
            noise_matrix_dim,
            noise_matrix_dim
        ]

        Args:
            time_series_squence (Tensor):
                a tensor of shape [batch_size, seq_len, 1]
        """

        src = self.encoder_input_layer(src)

        src = self.positional_encoding_layer(src)

        src = self.encoder(src=src)

        # feed encoder output to complex valued neural network

        # forward pass thorough complex valued CNN

        # return noise encoding matrices
        return src
