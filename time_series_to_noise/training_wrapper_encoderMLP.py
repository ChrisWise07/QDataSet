import torch
from utils import (
    load_Vo_dataset,
    calculate_ground_turth_parameters,
)
from constants import (
    DATA_SET_NAMES,
)

from encoder_with_MLP import EncoderWithMLP

from trace_based_loss import TraceBasedLoss

data1 = DATA_SET_NAMES[0]
data2 = data1 + "_D"  # distored control pulses
num_ex = 100  # note that file number 19 is corrupted and can't be loaded
num_epochs = 10

pulses, Vx, Vy, Vz = load_Vo_dataset(
    path_to_dataset=f"./QuantumDS/{data1}/{data2}",
    num_examples=num_ex,
)

ground_truth_parameters = torch.real(
    calculate_ground_turth_parameters(
        ground_truth_VX=Vx,
        ground_truth_VY=Vy,
        ground_truth_VZ=Vz,
    )
)

model = EncoderWithMLP(
    input_size=1024,
    d_model=512,
    n_heads=4,
    n_encoder_layers=2,
    num_noise_matrices=3,
    noise_matrix_dim=2,
    batch_first=True,
    dropout_encoder=0.1,
    dropout_pos_enc=0.1,
    dim_feedforward_encoder=2048,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = TraceBasedLoss()


for epoch in range(num_epochs):
    y_pred_parameters = model(pulses)

    loss = criterion(
        y_pred_parameters,
        VX_true=Vx,
        VY_true=Vy,
        VZ_true=Vz,
    )

    print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # if epoch % 10 == 0:
    #     print(f"Epoch: {epoch}, Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
