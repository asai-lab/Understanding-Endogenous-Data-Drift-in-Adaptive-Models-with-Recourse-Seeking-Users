import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset


class Recourse(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.action = nn.Parameter(pt.zeros(size))  
        # this mask is used to restrict which features can be changed
        # self.mask = pt.zeros(size)
        # self.mask[:, :17] = 1

    def forward(self, x: pt.Tensor, weight: pt.Tensor = None):
        # this mask is used to restrict which features can be changed
        # a = self.action * self.mask.detach()
        a = self.action
        x = x + a
        return_act = a.detach().clone()
        return x, return_act


def recourse(c_model: nn.Module, dataset: Dataset, max_epochs: int, weight: pt.Tensor = None, loss_list: list = None,cost_list = None,threshold = 1.0,q3RecourseCost: list = None,recourseModelLossList: list = None, isNew = None, new_cost_list = None, original_cost_list = None):
    loss: pt.Tensor
    r_model = Recourse(dataset.x.shape)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(r_model.parameters(), lr=0.5)
    weight = weight / weight.sum()

    r_model.train()
    for epoch in range(max_epochs):
        x_hat, return_act = r_model(dataset.x)
        y_hat = c_model(x_hat)
        
        # use bce loss to calculate loss(h(x'), threshold)
        target = pt.ones_like(y_hat)          
        target *= threshold
        bce_loss = criterion(y_hat, target)
        
        # action cost (squared error)
        action_cost = (x_hat - dataset.x) ** 2
        weighted_action_cost = (weight * action_cost).sum()

        # final loss function
        loss = bce_loss + 0.001 * weighted_action_cost

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(epoch == max_epochs - 1):
            print(f"Recourse Loss: {loss.item():.4f}")

    # save recourse result
    r_model.eval()
    recourse_result_x, result_action = r_model(dataset.x)
    score = c_model(recourse_result_x)
    recourse_result_y = (score > 0.5).float()
    
    print("Average Recourse Score:", score.mean().item())

    # calculate recourse cost
    if cost_list is not None:
        avgRecourseCost = 0.0
        avgOriginalRecourseCost = 0.0
        avgNewRecourseCost = 0.0
        newCount = 0

        sqr_action = result_action ** 2
        weighted_action_cost = (weight * sqr_action).sum(dim = 1)
        avgRecourseCost = weighted_action_cost.mean().item()
        for idx,t in enumerate(weighted_action_cost):
            if idx < isNew.size(0) and isNew[idx]:
                newCount += 1
                avgNewRecourseCost += t.item()
            else:
                avgOriginalRecourseCost += t.item()
        
        if newCount == 0:
            avgNewRecourseCost = 0.0
        else:
            avgNewRecourseCost /= newCount
        avgOriginalRecourseCost /= (len(weighted_action_cost) - newCount)
        cost_list.append(avgRecourseCost)
        new_cost_list.append(avgNewRecourseCost)
        original_cost_list.append(avgOriginalRecourseCost)


    # update dataset with recourse result
    dataset.y = recourse_result_y
    dataset.x = recourse_result_x.detach()
        
    return dataset, result_action.detach()