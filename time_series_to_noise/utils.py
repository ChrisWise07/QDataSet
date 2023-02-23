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


def trace_distance_batch_of_matrices(rho: Tensor, sigma: Tensor) -> Tensor:
    """
    Calculates the trace distance between two batches of density matrices.
    A factor of 1/2 is omitted for efficiency.

    Args:
        rho (np.ndarray): density matrix
        sigma (np.ndarray): density matrix

    Returns:
        Tensor: trace distance
    """
    return torch.norm(rho - sigma, p="fro", dim=(1, 2))


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
    return (
        trace_distance_batch_of_matrices(estimated_VX, VX_true).sum()
        + trace_distance_batch_of_matrices(estimated_VY, VY_true).sum()
        + trace_distance_batch_of_matrices(estimated_VZ, VZ_true).sum()
    ) / estimated_VX.shape[0]


def construct_estimated_VO_unital(
    psi: Tensor,
    theta: Tensor,
) -> Tensor:
    r"""
    Construct the estimated noise encoding matrix V_O from the
    parameters psi, theta, mu and the inverse of the pauli observable O.

    Args:
        psi (Tensor): parameter of shape (batch_size,)
        theta (Tensor): parameter of shape (batch_size,)

    Returns:
        Tensor: estimated noise encoding matrix QDQ^{\dagger} of shape (batch_size, 2, 2)
    """
    cos_2theta = torch.cos(2 * theta)
    sin_2theta = torch.sin(2 * theta)
    exp_2ipsi = torch.exp(2j * psi)
    exp_minus2ipsi = torch.exp(-2j * psi)

    return torch.stack(
        [
            cos_2theta,
            -exp_2ipsi * sin_2theta,
            -exp_minus2ipsi * sin_2theta,
            -cos_2theta,
        ],
        dim=-1,
    ).reshape(-1, 2, 2)


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
    (
        x_psi,
        x_theta,
        x_mu,
        y_psi,
        y_theta,
        y_mu,
        z_psi,
        z_theta,
        z_mu,
    ) = batch_parameters.T
    estimated_VX = construct_estimated_VO_unital(x_psi, x_theta)
    estimated_VY = construct_estimated_VO_unital(y_psi, y_theta)
    estimated_VZ = construct_estimated_VO_unital(z_psi, z_theta)
    return (
        SIGMA_X_DAGGER @ (estimated_VX * x_mu[:, None, None]),
        SIGMA_Y_DAGGER @ (estimated_VY * y_mu[:, None, None]),
        SIGMA_Z_DAGGER @ (estimated_VZ * z_mu[:, None, None]),
    )


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


def calculate_psi_theta_mu(matrix: Tensor) -> Tuple[float, float, float]:
    """
    Calculate the parameters psi, theta, mu from a noise encoding matrix.

    Args:
        matrix (Tensor): noise encoding matrix

    Returns:
        Tuple[float, float, float]: psi, theta, mu
    """
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)

    psi = (
        torch.log(eigenvectors[:, 0, 0]) - torch.log(eigenvectors[:, 1, 1])
    ) / 2j
    theta = torch.atan(eigenvectors[:, 0, 1] / eigenvectors[:, 0, 0])
    mu = eigenvalues[:, 0]
    return psi, theta, mu


def calculate_ground_turth_parameters(
    ground_truth_VX: Tensor,
    ground_truth_VY: Tensor,
    ground_truth_VZ: Tensor,
):
    """
    Calculate the ground truth parameters psi, theta, mu from the noise
    encoding matrices.

    Args:
        ground_truth_VX (Tensor): noise encoding matrix V_X
        ground_truth_VY (Tensor): noise encoding matrix V_Y
        ground_truth_VZ (Tensor): noise encoding matrix V_Z

    Returns:
        Tensor: ground truth parameters
    """
    X_psi, X_theta, X_mu = calculate_psi_theta_mu(
        SIGMA_X_DAGGER @ ground_truth_VX
    )
    Y_psi, Y_theta, Y_mu = calculate_psi_theta_mu(
        SIGMA_Y_DAGGER @ ground_truth_VY
    )
    Z_psi, Z_theta, Z_mu = calculate_psi_theta_mu(
        SIGMA_Z_DAGGER @ ground_truth_VZ
    )

    return torch.stack(
        (X_psi, X_theta, X_mu, Y_psi, Y_theta, Y_mu, Z_psi, Z_theta, Z_mu),
        dim=1,
    )
