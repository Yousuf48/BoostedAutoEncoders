from common import torch, data, datasets, transforms


# Creating custom datasets of pairs of noisy images and clean images for training (Gaussian noise)

class ImagesPair(data.Dataset):
    def __init__(self, dataset, mean=0.0, std=0.1):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        noise = image + torch.normal(mean=self.mean, std=self.std, size=image.shape)
        noisy_image = torch.clamp(noise, 0, 1)
        return image, noisy_image

    def __len__(self):
        return len(self.dataset)


training_dataset = ImagesPair(datasets.LFWPeople(root="data",
                                                 split='train',
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(64),
                                                     transforms.CenterCrop(64),
                                                     transforms.ToTensor(),

                                                 ])
                                                 ), 0.0, 0.5)

training_dataloader = data.DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = ImagesPair(datasets.LFWPeople(root="data",
                                             split='test',
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize(64),
                                                 transforms.CenterCrop(64),
                                                 transforms.ToTensor(),

                                             ])
                                             ), 0.0, 0.5)

test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
