import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
    
def training(model: nn.Module, dataset: Dataset, max_epochs: int, lr = 0.01):
    #data precessor
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    data_length = len(train_loader.dataset)

    loss: pt.Tensor
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr,weight_decay=0.001)
    epoch_loss = []
    
    model.train()
    for _ in range(max_epochs):
        running_loss = 0.0
        
        # devide data into several batch to train
        for X_batch,Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()

            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)


        train_loss = running_loss / data_length
        epoch_loss.append(train_loss)
        
    # print("MLP training loss: ", epoch_loss)
    