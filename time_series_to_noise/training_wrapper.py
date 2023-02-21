import torch
from utils import (
    load_Vo_dataset,
    trace_distance_based_loss,
    return_estimated_VO_unital_for_batch,
)
from constants import (
    DATA_SET_NAMES,
)
from EncoderWithMLP import EncoderWithMLP

data1 = DATA_SET_NAMES[0]
data2 = data1 + "_D"
dataset_name = data2  # "G_1q_XY_XZ_N1N5_D" # dataset name
num_ex = 9  # number of examples

pulses, Vx, Vy, Vz = load_Vo_dataset(
    path_to_dataset=f"./QuantumDS/{data1}/{data1}",
    num_examples=num_ex,
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    y_pred_parameters = model(pulses)

    (
        estimated_VX,
        estimated_VY,
        estimated_VZ,
    ) = return_estimated_VO_unital_for_batch(y_pred_parameters)

    loss = trace_distance_based_loss(
        estimated_VX, estimated_VY, estimated_VZ, Vx, Vy, Vz
    )

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
