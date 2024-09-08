from datasets import test_dataloader
from visualisation import show_images
from common import device, torch
from network import net

net.load_state_dict(torch.load('models/autoencoder_model.pth'))
samples = []
for i, (clean, degraded) in enumerate(test_dataloader):
    degraded = degraded.to(device)
    decode = net(degraded)

    samples.append(torch.cat((degraded[0].cpu(), decode[0].cpu()), dim=2))

    if i + 1 == 64:
        break

show_images(samples, "Model's Samples")
