import unittest
import torch
from time_series_to_noise.utils import (
    trace_distance,
    trace_distance_based_loss,
    construct_estimated_VO_unital,
    return_estimated_VO_unital_for_batch,
)

SIGMA_X = [[0, 1], [1, 0]]
SIGMA_Y = [[0, -1j], [1j, 0]]
SIGMA_Z = [[1, 0], [0, -1]]


class TestTUtilsTraceDistance(unittest.TestCase):
    def test_trace_distance_equal_matrices(self):
        rho = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        sigma = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        self.assertEqual(trace_distance(rho, sigma).item(), 0)

    def test_trace_distance_max(self):
        self.assertAlmostEqual(
            trace_distance(
                torch.tensor(SIGMA_X),
                torch.tensor(SIGMA_Y),
            ).item(),
            2.0,
            places=5,
        )

    def test_trace_distance_unequal_matrices(self):
        rho = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        sigma = torch.tensor([[0.75, 0.0], [0.0, 0.25]])
        self.assertAlmostEqual(
            trace_distance(rho, sigma).item(), 0.35355, places=5
        )


class TestTraceDistanceBasedLoss(unittest.TestCase):
    def test_trace_distance_based_loss_equal_matrcies(self):
        estimated_VX = torch.tensor([SIGMA_X] * 3, dtype=torch.complex64)
        estimated_VY = torch.tensor([SIGMA_Y] * 3, dtype=torch.complex64)
        estimated_VZ = torch.tensor([SIGMA_Z] * 3, dtype=torch.complex64)
        VX_true = estimated_VX.clone()
        VY_true = estimated_VY.clone()
        VZ_true = estimated_VZ.clone()
        self.assertEqual(
            trace_distance_based_loss(
                estimated_VX,
                estimated_VY,
                estimated_VZ,
                VX_true,
                VY_true,
                VZ_true,
            ).item(),
            0.0,
        )

    def test_trace_distance_based_loss_max(self):
        estimated_VX = torch.tensor([SIGMA_Y] * 3, dtype=torch.complex64)
        estimated_VY = torch.tensor([SIGMA_Z] * 3, dtype=torch.complex64)
        estimated_VZ = torch.tensor([SIGMA_X] * 3, dtype=torch.complex64)
        VX_true = torch.tensor([SIGMA_X] * 3, dtype=torch.complex64)
        VY_true = torch.tensor([SIGMA_Y] * 3, dtype=torch.complex64)
        VZ_true = torch.tensor([SIGMA_Z] * 3, dtype=torch.complex64)
        self.assertEqual(
            trace_distance_based_loss(
                estimated_VX,
                estimated_VY,
                estimated_VZ,
                VX_true,
                VY_true,
                VZ_true,
            ).item(),
            6.0,
        )

    def test_trace_distance_based_loss_mixture(self):
        estimated_VX = torch.tensor(
            [[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[1, 1], [1, 1]]],
            dtype=torch.complex64,
        )
        estimated_VY = torch.tensor(
            [[[0, 0], [0, 1]], [[1, 0], [0, 0]], [[1, 1], [1, 1]]],
            dtype=torch.complex64,
        )
        estimated_VZ = torch.tensor(
            [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 1], [1, 1]]],
            dtype=torch.complex64,
        )
        VX_true = torch.tensor(
            [[[0, 0], [0, 0]], [[0, 1], [1, 0]], [[1, 0], [0, 1]]],
            dtype=torch.complex64,
        )
        VY_true = torch.tensor(
            [[[0, 1], [1, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 1]]],
            dtype=torch.complex64,
        )
        VZ_true = torch.tensor(
            [[[0, 0], [0, 1]], [[1, 0], [0, 0]], [[1, 0], [0, 1]]],
            dtype=torch.complex64,
        )
        self.assertAlmostEqual(
            trace_distance_based_loss(
                estimated_VX,
                estimated_VY,
                estimated_VZ,
                VX_true,
                VY_true,
                VZ_true,
            ).item(),
            4.52835,
            places=5,
        )


class TestConstructEstimatedVOUnital(unittest.TestCase):
    def test_construct_estimated_VO_unital_sigma_x(self):
        O_dagger = torch.tensor(SIGMA_X, dtype=torch.complex64)
        psi = torch.tensor(0.5)
        theta = torch.tensor(0.25)
        mu = torch.tensor(1.0)
        expected_V = torch.tensor(
            [
                [-0.2590 + 0.4034j, -0.8776 + 0.0000j],
                [0.8776 + 0.0000j, -0.2590 - 0.4034j],
            ],
            dtype=torch.complex64,
        )
        self.assertTrue(
            torch.allclose(
                construct_estimated_VO_unital(psi, theta, mu, O_dagger),
                expected_V,
                atol=1e-04,
            )
        )

    def test_construct_estimated_VO_unital_sigma_y(self):
        O_dagger = torch.tensor(SIGMA_Y, dtype=torch.complex64)
        psi = torch.tensor(0.1)
        theta = torch.tensor(0.3)
        mu = torch.tensor(0.5)
        expected_V = torch.tensor(
            [
                [0.0561 + 0.2767j, 0.0000 + 0.4127j],
                [0.0000 + 0.4127j, 0.0561 - 0.2767j],
            ],
            dtype=torch.complex64,
        )
        self.assertTrue(
            torch.allclose(
                construct_estimated_VO_unital(psi, theta, mu, O_dagger),
                expected_V,
                atol=1e-04,
            )
        )

    def test_construct_estimated_VO_unital_sigma_z(self):
        O_dagger = torch.tensor(SIGMA_Z, dtype=torch.complex64)
        psi = torch.tensor(0.2)
        theta = torch.tensor(0.4)
        mu = torch.tensor(0.8)
        expected_V = torch.tensor(
            [
                [0.5574 + 0.0000j, -0.5286 - 0.2235j],
                [0.5286 - 0.2235j, 0.5574 + 0.00000j],
            ],
            dtype=torch.complex64,
        )
        self.assertTrue(
            torch.allclose(
                construct_estimated_VO_unital(psi, theta, mu, O_dagger),
                expected_V,
                atol=1e-04,
            )
        )


class TestReturnEstimatedVOUnitalForBatch(unittest.TestCase):
    def test_return_estimated_VO_unital_for_batch_1(self):
        batch_parameters = torch.tensor(
            [
                [0.5, 0.25, 1.0, 0.1, 0.3, 0.5, 0.2, 0.4, 0.8],
                [
                    0.2,
                    0.4,
                    0.8,
                    0.5,
                    0.25,
                    1.0,
                    0.1,
                    0.3,
                    0.5,
                ],
            ],
            dtype=torch.float32,
        )
        (
            estimated_VX,
            estimated_VY,
            estimated_VZ,
        ) = return_estimated_VO_unital_for_batch(batch_parameters)

        expected_VX = torch.tensor(
            [
                [
                    [-0.2590 + 0.4034j, -0.8776 + 0.0000j],
                    [0.8776 + 0.0000j, -0.2590 - 0.4034j],
                ],
                [
                    [-0.5286 + 0.2235j, -0.5574 + 0.00000j],
                    [0.5574 + 0.0000j, -0.5286 - 0.2235j],
                ],
            ]
        )
        expected_VY = torch.tensor(
            [
                [
                    [0.0561 + 0.2767j, 0.0000 + 0.4127j],
                    [0.0000 + 0.4127j, 0.0561 - 0.2767j],
                ],
                [
                    [0.4034 + 0.259j, 0.0000 + 0.8776j],
                    [0.0000 + 0.8776j, 0.4034 - 0.2590j],
                ],
            ]
        )

        expected_VZ = torch.tensor(
            [
                [
                    [0.5574 + 0.0000j, -0.5286 - 0.2235j],
                    [0.5286 - 0.2235j, 0.5574 + 0.00000j],
                ],
                [
                    [0.4127 + 0.0000j, -0.2767 - 0.0561j],
                    [0.2767 - 0.0561j, 0.4127 + 0.0000j],
                ],
            ]
        )
        self.assertEqual(estimated_VX.shape, (2, 2, 2))
        self.assertEqual(estimated_VY.shape, (2, 2, 2))
        self.assertEqual(estimated_VZ.shape, (2, 2, 2))
        self.assertTrue(
            torch.allclose(
                estimated_VX,
                expected_VX,
                atol=1e-04,
            )
        )
        self.assertTrue(
            torch.allclose(
                estimated_VY,
                expected_VY,
                atol=1e-04,
            )
        )
        self.assertTrue(
            torch.allclose(
                estimated_VZ,
                expected_VZ,
                atol=1e-04,
            )
        )
