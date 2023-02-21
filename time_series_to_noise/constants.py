import torch

DATA_SET_NAMES = [
    "G_1q_X",
    "G_1q_XY",
    "G_1q_XY_XZ_N1N5",
    "G_1q_XY_XZ_N1N6",
    "G_1q_XY_XZ_N3N6",
    "G_1q_X_Z_N1",
    "G_1q_X_Z_N2",
    "G_1q_X_Z_N3",
    "G_1q_X_Z_N4",
    "G_2q_IX-XI_IZ-ZI_N1-N6",
    "G_2q_IX-XI-XX",
    "G_2q_IX-XI-XX_IZ-ZI_N1-N5",
    "G_2q_IX-XI-XX_IZ-ZI_N1-N5",
    "S_1q_X",
    "S_1q_XY",
    "S_1q_XY_XZ_N1N5",
    "S_1q_XY_XZ_N1N6",
    "S_1q_XY_XZ_N3N6",
    "S_1q_X_Z_N1",
    "S_1q_X_Z_N2",
    "S_1q_X_Z_N3",
    "S_1q_X_Z_N4",
    "S_2q_IX-XI_IZ-ZI_N1-N6",
    "S_2q_IX-XI-XX",
    "S_2q_IX-XI-XX_IZ-ZI_N1-N5",
    "S_2q_IX-XI-XX_IZ-ZI_N1-N6",
]

SIGMA_X_DAGGER = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
SIGMA_Y_DAGGER = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
SIGMA_Z_DAGGER = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
