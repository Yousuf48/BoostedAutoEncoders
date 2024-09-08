from common import nn, optim, device, torch, np
from datasets import training_dataloader
from visualisation import show_images
from network import net
import time

images, labels = next(iter(training_dataloader))
show_images(images, "Some Training Images")

epochs = 150

# Mean Square Error (MSE)
mse = nn.MSELoss()

optimiser = optim.Adam(net.parameters(), lr=0.004)

start_time = time.time()

for i in range(epochs):
    losses = []
    net.train()
    for j, (data) in enumerate(training_dataloader):
        clean_images, noisy_images = data

        clean_images, noisy_images = clean_images.to(device), noisy_images.to(device)

        # training process
        net.zero_grad()
        output = net(noisy_images)
        loss = mse(output, clean_images)
        loss.backward()
        optimiser.step()

        losses.append(loss.item())

    if i + 1 == 40:
        optimiser = optim.Adam(net.parameters(), lr=0.003)
    if i + 1 == 95:
        optimiser = optim.Adam(net.parameters(), lr=0.002)
    if i + 1 == 125:
        optimiser = optim.Adam(net.parameters(), lr=0.001)

    print(f"Epoch {i}: the average loss of the epoch\033[1m {np.mean(losses)}\033[0m.")

total_time = time.time() - start_time
print(f"The total training time is {total_time / 60} min, each epoch took {(total_time / epochs) / 60} min.")
torch.save(net.state_dict(), "models/boostedAEs.pth")
