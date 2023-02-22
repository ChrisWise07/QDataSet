import torch
from utils import (
    load_Vo_dataset,
    trace_distance_based_loss,
    return_estimated_VO_unital_for_batch,
)
from constants import (
    DATA_SET_NAMES,
)
from SimpleANN import SimpleANN

data1 = DATA_SET_NAMES[0]
data2 = data1 + "_D"  # distored control pulses
num_ex = 3  # note that file number 19 is corrupted and can't be loaded
num_epochs = 3

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    y_pred_parameters = model(pulses)
    print(f"y_pred_parameters.shape:\n {y_pred_parameters.shape}")
    print(f"y_pred_parameters:\n {y_pred_parameters}")

    (
        estimated_VX,
        estimated_VY,
        estimated_VZ,
    ) = return_estimated_VO_unital_for_batch(y_pred_parameters)

    print(f"estimated_VX:\n {estimated_VX}")
    print(f"estimated_VY:\n {estimated_VY}")
    print(f"estimated_VZ:\n {estimated_VZ}")

    loss = trace_distance_based_loss(
        estimated_VX, estimated_VY, estimated_VZ, Vx, Vy, Vz
    )

    print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # if epoch % 10 == 0:
    #     print(f"Epoch: {epoch}, Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
