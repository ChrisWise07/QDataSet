import torch
from torch import Tensor
import zipfile
import pickle
import numpy as np
from typing import Tuple

from constants import (
    SIGMA_X_DAGGER,
    SIGMA_Y_DAGGER,
    SIGMA_Z_DAGGER,
)


def trace_distance(rho: Tensor, sigma: Tensor) -> Tensor:
    """
    Calculates the trace distance between two density matrices. A factor
    of 1/2 is omitted for efficiency.

    Args:
        rho (np.ndarray): density matrix
        sigma (np.ndarray): density matrix

    Returns:
        Tensor: trace distance
    """
    return torch.norm(rho - sigma, p="fro")


def trace_distance_based_loss(
    estimated_VX: Tensor,
    estimated_VY: Tensor,
    estimated_VZ: Tensor,
    VX_true: Tensor,
    VY_true: Tensor,
    VZ_true: Tensor,
) -> Tensor:
    """
    Calculate the squared l2 norm of the difference between the
    predicted and true noise matrices.

    Args:
        predicted_noise_matrices (Tensor): predicted noise matrices
        true_noise_matrices (Tensor): true noise matrices

    Returns:
        Tensor: loss
    """

    num_matrices = estimated_VX.shape[0]
    loss = torch.tensor(0.0)
    for i in range(num_matrices):
        loss += (
            trace_distance(estimated_VX[i], VX_true[i])
            + trace_distance(estimated_VY[i], VY_true[i])
            + trace_distance(estimated_VZ[i], VZ_true[i])
        )
    print(f"loss in trace based loss function: {loss}")
    return (loss / num_matrices).clone().detach().requires_grad_(True)


def construct_estimated_VO_unital(
    psi: float, theta: float, mu: float, O_dagger: Tensor
) -> Tensor:
    """
    Construct the estimated noise encoding matrix V_O from the
    parameters psi, theta, mu and the inverse of the pauli observable O.

    Args:
        psi (float): parameter
        theta (float): parameter
        mu (float): parameter
        O_inv (Tensor): inverse of the pauli observable

    Returns:
        Tensor: estimated noise encoding matrix V_O
    """
    cos_2theta = torch.cos(2 * theta)
    sin_2theta = torch.sin(2 * theta)
    exp_2ipsi = torch.exp(2j * psi)
    exp_minus2ipsi = torch.exp(-2j * psi)

    return torch.matmul(
        O_dagger,
        torch.tensor(
            [
                [mu * cos_2theta, -exp_2ipsi * mu * sin_2theta],
                [-exp_minus2ipsi * mu * sin_2theta, -mu * cos_2theta],
            ]
        ),
    )


def return_estimated_VO_unital_for_batch(
    batch_parameters: Tensor,
) -> Tensor:
    """
    Construct the estimated noise encoding matrices V_O for a batch of
    parameters.

    Args:
        batch_parameters (Tensor): batch of parameters

    Returns:
        Tensor: estimated noise encoding matrices V_O
    """
    batch_size = batch_parameters.shape[0]
    estimated_VX = torch.zeros((batch_size, 2, 2), dtype=torch.complex64)
    estimated_VY = torch.zeros((batch_size, 2, 2), dtype=torch.complex64)
    estimated_VZ = torch.zeros((batch_size, 2, 2), dtype=torch.complex64)
    for i in range(batch_size):
        estimated_VX[i] = construct_estimated_VO_unital(
            batch_parameters[i, 0],
            batch_parameters[i, 1],
            batch_parameters[i, 2],
            SIGMA_X_DAGGER,
        )
        estimated_VY[i] = construct_estimated_VO_unital(
            batch_parameters[i, 3],
            batch_parameters[i, 4],
            batch_parameters[i, 5],
            SIGMA_Y_DAGGER,
        )
        estimated_VZ[i] = construct_estimated_VO_unital(
            batch_parameters[i, 6],
            batch_parameters[i, 7],
            batch_parameters[i, 8],
            SIGMA_Z_DAGGER,
        )
    return estimated_VX, estimated_VY, estimated_VZ


def load_Vo_dataset(
    path_to_dataset: str, num_examples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset of pulses and Vo operators.

    Args:
        dataset_name: Name of the dataset.
        num_examples: Number of examples to load.

    Returns:
        pulses: Array of pulses.
        Vx: Array of X Vo operators.
        Vy: Array of Y Vo operators.
        Vz: Array of Z Vo operators.
    """
    pulses = np.zeros((num_examples, 1024), dtype=np.float32)
    Vx = np.zeros((num_examples, 2, 2), dtype=np.complex64)
    Vy = np.zeros((num_examples, 2, 2), dtype=np.complex64)
    Vz = np.zeros((num_examples, 2, 2), dtype=np.complex64)

    with zipfile.ZipFile(f"{path_to_dataset}.zip", mode="r") as fzip:
        for index, fname in enumerate(fzip.namelist()[:num_examples]):
            with fzip.open(fname, "r") as f:
                # print(f"Loading {fname}...")
                data = pickle.load(f)
                pulses[index, :] = data["pulses"][0, :, 0].reshape(
                    1024,
                )
                Vx[index], Vy[index], Vz[index] = data["Vo_operator"]
    return (
        torch.from_numpy(pulses),
        torch.from_numpy(Vx),
        torch.from_numpy(Vy),
        torch.from_numpy(Vz),
    )
