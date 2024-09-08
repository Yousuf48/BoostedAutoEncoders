from common import torch, nn, device

print(f"Device: {device}")


class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Autoencoder, self).__init__()
        self.channels_num = in_channels
        self.feature_maps = out_channels
        self.encoders = nn.ModuleList([self.create_encoder() for _ in range(5)])
        self.decoder = self.create_decoder()

        self.flatten = nn.Flatten()
        self.latent_space = nn.Sequential(
            nn.Linear(self.feature_maps * 16 * 4 * 4, 4096), nn.ReLU(),  # (16384) -> (4096)
            nn.Linear(4096, self.feature_maps * 16 * 4 * 4), nn.ReLU(),  # (4096)  -> (16384)

        )

    def create_encoder(self):
        return nn.Sequential(
            nn.Conv2d(self.channels_num, self.feature_maps, 4, 2, 1), nn.BatchNorm2d(self.feature_maps), nn.ReLU(),
            # (3,64,64)   --> (64,32,32)
            nn.Conv2d(self.feature_maps, self.feature_maps, 3, 1, 1), nn.BatchNorm2d(self.feature_maps), nn.ReLU(),
            # (64,32,32)  --> (64,32,32)
            nn.Conv2d(self.feature_maps, self.feature_maps, 3, 1, 1), nn.BatchNorm2d(self.feature_maps), nn.ReLU(),
            # (64,32,32)  --> (128,32,32)
            nn.Conv2d(self.feature_maps, self.feature_maps, 3, 1, 1), nn.BatchNorm2d(self.feature_maps), nn.ReLU(),
            # (64,32,32)  --> (64,32,32)
            nn.Conv2d(self.feature_maps, self.feature_maps * 2, 3, 1, 1), nn.BatchNorm2d(self.feature_maps * 2),
            nn.ReLU(),  # (128,32,32)  --> (128,32,32)
            nn.Conv2d(self.feature_maps * 2, self.feature_maps * 2, 3, 1, 1), nn.BatchNorm2d(self.feature_maps * 2),
            nn.ReLU(),  # (128,32,32)  --> (128,32,32)
            nn.Conv2d(self.feature_maps * 2, self.feature_maps * 2, 3, 1, 1), nn.BatchNorm2d(self.feature_maps * 2),
            nn.ReLU(),  # (128,32,32)  --> (128,32,32)
            nn.Conv2d(self.feature_maps * 2, self.feature_maps * 2, 3, 1, 1), nn.BatchNorm2d(self.feature_maps * 2),
            nn.ReLU(),  # (128,32,32)  --> (128,32,32)
            nn.Conv2d(self.feature_maps * 2, self.feature_maps * 4, 4, 2, 1), nn.BatchNorm2d(self.feature_maps * 4),
            nn.ReLU(),  # (128,32,32)  --> (256,16,16)
            nn.Conv2d(self.feature_maps * 4, self.feature_maps * 8, 4, 2, 1), nn.BatchNorm2d(self.feature_maps * 8),
            nn.ReLU(),  # (256,16,16) --> (512,8,8)
            nn.Conv2d(self.feature_maps * 8, self.feature_maps * 16, 4, 2, 1),  # (512,8,8)   --> (1024,4,4)
        )

    def create_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.feature_maps * 16, self.feature_maps * 8, 4, 2, 1),
            nn.BatchNorm2d(self.feature_maps * 8), nn.ReLU(),  # (1024,4,4)   --> (512,8,8)
            nn.ConvTranspose2d(self.feature_maps * 8, self.feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(self.feature_maps * 4), nn.ReLU(),  # (512,8,8)   --> (256,16,16)
            nn.ConvTranspose2d(self.feature_maps * 4, self.feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(self.feature_maps * 4), nn.ReLU(),  # (256,16,16) --> (256,32,32)
            nn.ConvTranspose2d(self.feature_maps * 4, self.feature_maps * 2, 3, 1, 1),
            nn.BatchNorm2d(self.feature_maps * 2), nn.ReLU(),  # (256,32,32)  --> (128,32,32)
            nn.ConvTranspose2d(self.feature_maps * 2, self.feature_maps * 2, 3, 1, 1),
            nn.BatchNorm2d(self.feature_maps * 2), nn.ReLU(),  # (128,32,32)  --> (128,32,32)
            nn.ConvTranspose2d(self.feature_maps * 2, self.feature_maps * 2, 3, 1, 1),
            nn.BatchNorm2d(self.feature_maps * 2), nn.ReLU(),  # (128,32,32)  --> (128,32,32)
            nn.ConvTranspose2d(self.feature_maps * 2, self.feature_maps * 2, 3, 1, 1),
            nn.BatchNorm2d(self.feature_maps * 2), nn.ReLU(),  # (128,32,32)  --> (128,32,32)
            nn.ConvTranspose2d(self.feature_maps * 2, self.feature_maps, 3, 1, 1), nn.BatchNorm2d(self.feature_maps),
            nn.ReLU(),  # (128,32,32)  --> (64,32,32)
            nn.ConvTranspose2d(self.feature_maps, self.feature_maps, 3, 1, 1), nn.BatchNorm2d(self.feature_maps),
            nn.ReLU(),  # (64,32,32)  --> (64,32,32)
            nn.ConvTranspose2d(self.feature_maps, self.feature_maps, 3, 1, 1), nn.BatchNorm2d(self.feature_maps),
            nn.ReLU(),  # (64,32,32)  --> (64,32,32)
            nn.ConvTranspose2d(self.feature_maps, self.channels_num, 4, 2, 1),  # (64,32,32)  --> (3,64,64)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = [encoder(x) for encoder in self.encoders]

        encoded_average = torch.mean(torch.stack((encoded)), dim=0)
        latent_space = self.latent_space(self.flatten(encoded_average))
        unflatten = latent_space.view(-1, 1024, 4, 4)
        decode = self.decoder(unflatten)
        return decode


net = Autoencoder(3, 64).to(device)
