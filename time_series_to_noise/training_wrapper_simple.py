import torch
from utils import (
    load_Vo_dataset,
    trace_distance_based_loss,
    return_estimated_VO_unital_for_batch,
)
from constants import (
    DATA_SET_NAMES,
)
from SimpleModel import SimpleANN

data1 = DATA_SET_NAMES[0]
data2 = data1 + "_D"
dataset_name = data2  # "G_1q_XY_XZ_N1N5_D" # dataset name
num_ex = 2  # number of examples

pulses, Vx, Vy, Vz = load_Vo_dataset(
    path_to_dataset=f"./QuantumDS/{data1}/{data2}",
    num_examples=num_ex,
)
print(f"Shape of input squence:\n {pulses.shape}")
print(f"Shape of ground truth V_O:\n {Vx.shape}, {Vy.shape}, {Vz.shape}")


model = SimpleANN(
    input_size=1024,
    num_noise_matrices=3,
    noise_matrix_dim=2,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred_parameters = model(pulses)
    # print(f"y_pred_parameters.shape:\n {y_pred_parameters.shape}")
    # print(f"y_pred_parameters:\n {y_pred_parameters}")

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
