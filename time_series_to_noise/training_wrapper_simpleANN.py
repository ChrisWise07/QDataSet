import torch
import torch.nn as nn
from utils import (
    load_Vo_dataset,
    calculate_ground_turth_parameters,
    VO_parameter_estimation_loss_wrapper,
)
from constants import (
    DATA_SET_NAMES,
)

from SimpleANN import SimpleANN

data1 = DATA_SET_NAMES[0]
data2 = data1 + "_D"  # distored control pulses
num_ex = 40  # note that file number 19 is corrupted and can't be loaded
num_epochs = 2

pulses, Vx, Vy, Vz = load_Vo_dataset(
    path_to_dataset=f"./QuantumDS/{data1}/{data2}",
    num_examples=num_ex,
)
# print(f"Shape of input squence:\n {pulses.shape}")
# print(f"Shape of ground truth V_O:\n {Vx.shape}, {Vy.shape}, {Vz.shape}")

ground_truth_parameters = torch.real(
    calculate_ground_turth_parameters(
        ground_truth_VX=Vx,
        ground_truth_VY=Vy,
        ground_truth_VZ=Vz,
    )
)


model = SimpleANN(
    input_size=1024,
    num_noise_matrices=3,
    noise_matrix_dim=2,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = VO_parameter_estimation_loss_wrapper


for epoch in range(num_epochs):
    y_pred_parameters = model(pulses)

    loss = criterion(
        y_pred_parameters=y_pred_parameters,
        y_true_VX=Vx,
        y_true_VY=Vy,
        y_true_VZ=Vz,
    )
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    print(type(loss))
    print(loss)

    # if epoch % 10 == 0:
    #     print(f"Epoch: {epoch}, Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
