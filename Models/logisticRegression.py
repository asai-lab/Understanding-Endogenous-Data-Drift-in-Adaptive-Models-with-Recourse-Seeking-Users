import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
def training(model: nn.Module, dataset: Dataset, max_epochs: int, lr = 0.01):
    #data precessor
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    data_Length = len(train_loader.dataset)

    loss: pt.Tensor
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr,weight_decay=0.001)
    epoch_loss = []
    
    model.train()
    for _ in range(max_epochs):
        running_loss = 0.0

        # divide data into several batch to train
        for X_batch,Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            # if dimension of outputs != dimension of Y_batch, then unsqueeze outputs
            if outputs.dim() != Y_batch.dim():
                outputs = outputs.unsqueeze(-1)
           
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            # print("logistic Regression loss: ",loss)
            
            
        train_loss = running_loss / data_Length
        epoch_loss.append(train_loss)
    
    # print("Logistic Regression training loss: ", epoch_loss)
        
    