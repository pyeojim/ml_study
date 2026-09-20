import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import matplotlib.pyplot as plt

BS = 128

class cifarDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform
        files = [f"data_batch_{i}" for i in range(1, 6)]
        for x in files:
            path = os.path.join(img_dir, x)
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            self.data.append(torch.from_numpy(batch[b"data"]))
            self.labels.extend(batch[b"labels"])
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].float() / 255.0
        label = self.labels[idx]

        return img, label

class neuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main(epochs):
    data = cifarDataset('cifar-10')
    train_data, test_data = random_split(data, [int(len(data)*0.8), len(data) - int(len(data)*0.8)])
    train_dataloader = DataLoader(train_data, batch_size=BS, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BS, shuffle=False)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = neuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):    
        train_stat = [0,0,0]
        test_stat = [0,0,0]

        model.train()

        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = loss_func(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)            
            train_stat[0] += loss.item() * y.size(0)
            train_stat[1] += (pred == y).sum().item()
            train_stat[2] += y.size(0)

        model.eval()

        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                loss = loss_func(logits, y)

                pred = logits.argmax(dim=1)
                test_stat[0] += loss.item() * y.size(0)
                test_stat[1] += (pred == y).sum().item()
                test_stat[2] += y.size(0)

        print(f"epoch {epoch+1} train loss: {(train_stat[0] / train_stat[2]):.4f}, train accuracy: {(train_stat[1]/train_stat[2]):.4f}, test loss: {(test_stat[0] / test_stat[2]):.4f}, test accuracy: {test_stat[1] / test_stat[2]:.4f}")
            
main(100)

# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].reshape(3,32,32).permute(1,2,0)
# label = train_labels[0]
# plt.imshow(img)
# plt.show()
# print(f"Label: {label}")