from torch import Tensor, nn

from utils import (
    return_estimated_VO_unital_for_batch,
    trace_distance_based_loss,
)


class TraceBasedLoss(nn.Module):
    def __init__(self):
        super(TraceBasedLoss, self).__init__()

    def forward(self, y_pred_parameters, VX_true, VY_true, VZ_true) -> Tensor:
        """
        Compute the loss between the estimated noise matrices and the
        true noise matrices.

        Args:
            y_pred_parameters (Tensor): predicted parameters
            VX_true (Tensor): true noise encoding matrix V_X
            VY_true (Tensor): true noise encoding matrix V_Y
            VZ_true (Tensor): true noise encoding matrix V_Z

        Returns:
            Tensor: loss
        """
        (
            estimated_VX,
            estimated_VY,
            estimated_VZ,
        ) = return_estimated_VO_unital_for_batch(y_pred_parameters)

        loss = trace_distance_based_loss(
            estimated_VX,
            estimated_VY,
            estimated_VZ,
            VX_true,
            VY_true,
            VZ_true,
        )

        # Compute the batch loss
        return loss / y_pred_parameters.size(0)
