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


def VO_parameter_estimation_loss_wrapper(
    y_pred_parameters, y_true_VX, y_true_VY, y_true_VZ
):
    """
    Wrapper function for the loss function that calculates the loss
    between the estimated noise matrices and the true noise matrices.

    Args:
        y_pred_parameters (Tensor): predicted parameters
        y_true_VX (Tensor): true noise encoding matrix V_X
        y_true_VY (Tensor): true noise encoding matrix V_Y
        y_true_VZ (Tensor): true noise encoding matrix V_Z

    Returns:
        Tensor: loss
    """
    (
        estimated_VX,
        estimated_VY,
        estimated_VZ,
    ) = return_estimated_VO_unital_for_batch(y_pred_parameters)
    return trace_distance_based_loss(
        estimated_VX,
        estimated_VY,
        estimated_VZ,
        y_true_VX,
        y_true_VY,
        y_true_VZ,
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


def calculate_ground_turth_parameters(
    ground_truth_VX: Tensor,
    ground_truth_VY: Tensor,
    ground_truth_VZ: Tensor,
):
    batch_size = ground_truth_VX.shape[0]
    ground_turth_parameters = torch.zeros(
        (batch_size, 9), dtype=torch.complex64
    )
    for i in range(batch_size):
        X_eigenvalues, X_eigenvectors = torch.linalg.eig(ground_truth_VX[i])
        X_mu = X_eigenvalues[0].item()
        X_psi = (
            torch.log(X_eigenvectors[0, 0]) - torch.log(X_eigenvectors[1, 1])
        ) / 2j
        X_theta = torch.atan(X_eigenvectors[0, 1] / X_eigenvectors[0, 0])
        Y_eigenvalues, Y_eigenvectors = torch.linalg.eig(ground_truth_VY[i])
        Y_mu = Y_eigenvalues[0].item()
        Y_psi = (
            torch.log(Y_eigenvectors[0, 0]) - torch.log(Y_eigenvectors[1, 1])
        ) / 2j
        Y_theta = torch.atan(Y_eigenvectors[0, 1] / Y_eigenvectors[0, 0])
        Z_eigenvalues, Z_eigenvectors = torch.linalg.eig(ground_truth_VZ[i])
        Z_mu = Z_eigenvalues[0].item()
        Z_psi = (
            torch.log(Z_eigenvectors[0, 0]) - torch.log(Z_eigenvectors[1, 1])
        ) / 2j
        Z_theta = torch.atan(Z_eigenvectors[0, 1] / Z_eigenvectors[0, 0])
        ground_turth_parameters[i] = torch.tensor(
            [
                X_psi,
                X_theta,
                X_mu,
                Y_psi,
                Y_theta,
                Y_mu,
                Z_psi,
                Z_theta,
                Z_mu,
            ]
        )
    return ground_turth_parameters


def calculate_ground_turth_parameters_updated(
    ground_truth_VX: Tensor,
    ground_truth_VY: Tensor,
    ground_truth_VZ: Tensor,
):
    batch_size = ground_truth_VX.shape[0]
    ground_turth_parameters = torch.zeros(
        (batch_size, 9), dtype=torch.complex64
    )

    X_eigenvalues, X_eigenvectors = torch.linalg.eig(ground_truth_VX)
    X_mu = X_eigenvalues[:, 0].reshape(-1, 1)
    X_psi = (
        torch.log(X_eigenvectors[:, 0, 0]) - torch.log(X_eigenvectors[:, 1, 1])
    ) / 2j
    X_theta = torch.atan(
        X_eigenvectors[:, 0, 1] / X_eigenvectors[:, 0, 0]
    ).reshape(-1, 1)

    Y_eigenvalues, Y_eigenvectors = torch.linalg.eig(ground_truth_VY)
    Y_mu = Y_eigenvalues[:, 0].reshape(-1, 1)
    Y_psi = (
        torch.log(Y_eigenvectors[:, 0, 0]) - torch.log(Y_eigenvectors[:, 1, 1])
    ) / 2j
    Y_theta = torch.atan(
        Y_eigenvectors[:, 0, 1] / Y_eigenvectors[:, 0, 0]
    ).reshape(-1, 1)

    Z_eigenvalues, Z_eigenvectors = torch.linalg.eig(ground_truth_VZ)
    Z_mu = Z_eigenvalues[:, 0].reshape(-1, 1)
    Z_psi = (
        torch.log(Z_eigenvectors[:, 0, 0]) - torch.log(Z_eigenvectors[:, 1, 1])
    ) / 2j
    Z_theta = torch.atan(
        Z_eigenvectors[:, 0, 1] / Z_eigenvectors[:, 0, 0]
    ).reshape(-1, 1)

    ground_turth_parameters[:, :3] = torch.cat([X_psi, X_theta, X_mu], dim=0)
    ground_turth_parameters[:, 3:6] = torch.cat([Y_psi, Y_theta, Y_mu], dim=0)
    ground_turth_parameters[:, 6:] = torch.cat([Z_psi, Z_theta, Z_mu], dim=0)

    return ground_turth_parameters
