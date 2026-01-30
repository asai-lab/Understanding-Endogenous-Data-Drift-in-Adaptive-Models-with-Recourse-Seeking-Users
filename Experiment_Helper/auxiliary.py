import torch as pt
import numpy as np
import math
import pandas as pd
import datetime
from scipy.stats import invgamma

def getFunc(x):
    # Function definition
    return math.log(x - 0.9) + 2.5
    
def getWeights(feature_num, type = 'uniform'):
    # type can be chosen from uniform, log
    if(type == 'uniform'):
        return pt.from_numpy(np.ones(feature_num)) / feature_num
    elif(type == 'log'):
        weights = np.array([getFunc(i) for i in range(1, feature_num + 1)])
        weights = weights / np.sum(weights)  # Normalize the weights
        return pt.from_numpy(weights)
    elif(type == 'inverse_gamma'):
        alpha = 3.0  # shape parameter
        beta = 1.0   # scale parameter
        x_values = np.linspace(0.1, 5.0, feature_num)
        
        weights = invgamma.pdf(x_values, a=alpha, scale=beta)
        weights = weights / np.sum(weights)  # Normalize the weights
        return pt.from_numpy(weights)
        
    else:
        print("type is incorrect at getWeights")

def update_train_data(train, sample, model, type = 'all', expected_size = None):
    """
    This function updates the training dataset by incorporating samples from sample dataset.
    The type parameter determines how the update is performed:
    - 'none': No update is performed; the original training dataset is returned.
    - 'all': The training dataset is completely replaced with samples from the sample dataset.
    - 'mixed': Half of the training dataset is retained, and the other half is replaced with samples from the sample dataset.
    """
    #type can be chosen from none, all, mixed
    if type == 'none':
        return train, pt.empty(0, dtype=pt.bool)


    if (expected_size is not None) and (expected_size > train.x.shape[0]):
        size = expected_size
    else:
        size = train.x.shape[0]

    sample_indices = pt.randperm(sample.x.shape[0])[:size]
    sampled_x = sample.x[sample_indices]

    if type == 'mixed':
        # the number of samples to keep and to add
        num_to_retain = train.x.shape[0] // 2
        num_to_add = size - num_to_retain
        if num_to_add < 0: 
            num_to_add = 0
            num_to_retain = size
        
        # get the data that will be retained
        retain_indices_from_old = pt.randperm(train.x.shape[0])[:num_to_retain]
        retained_x = train.x[retain_indices_from_old]
        retained_y = train.y[retain_indices_from_old]

        # get the new data that will be added
        new_x_from_sample = sampled_x[:num_to_add]
        with pt.no_grad():
            y_prob_new = model(new_x_from_sample).squeeze(1)
            new_y_from_sample = pt.where(y_prob_new > 0.5, 1.0, 0.0) # pseudo-labels for the new data using the model
        
        # combine the retained and new data
        final_x = pt.cat([retained_x, new_x_from_sample], dim=0)
        final_y = pt.cat([retained_y, new_y_from_sample], dim=0)

        # create a boolean mask to track which samples are new
        isNew = pt.zeros(final_x.shape[0], dtype=pt.bool)
        isNew[num_to_retain:] = True # The second part of the tensor is new.

        # shuffle the combined dataset
        shuffled_indices = pt.randperm(final_x.shape[0])
        train.x = final_x[shuffled_indices]
        train.y = final_y[shuffled_indices]
        isNew = isNew[shuffled_indices]


        num_zeros = (train.y == 0).sum().item()
        num_ones = (train.y == 1).sum().item()
        print(f"Number of 0s: {num_zeros}")
        print(f"Number of 1s: {num_ones}")

        return train, isNew


    if type == 'all':
        train.x = sampled_x
        with pt.no_grad():
            y_prob: pt.Tensor = model(train.x)
        y_prob = y_prob.squeeze(1)
        train.y = pt.where(y_prob > 0.5, 1.0, 0.0)

        return train, pt.empty(0, dtype=pt.bool)

    num_zeros = (train.y == 0).sum().item()
    num_ones = (train.y == 1).sum().item()
    print(f"Number of 0s: {num_zeros}")
    print(f"Number of 1s: {num_ones}")

class FileSaver:
    def __init__(self, fail_to_recourse, avgRecourseCost_list, avgNewRecourseCostList, avgOriginalRecourseCostList, t_rate_list, failToRecourse_old, failToRecourse_new, avg_score_list, balanced_acc_list):
        # Initialize the attributes with the provided lists
        self.failToRecourse = fail_to_recourse
        self.avgRecourseCost = avgRecourseCost_list
        self.avgNewRecourseCost = avgNewRecourseCostList
        self.avgOriginalRecourseCost = avgOriginalRecourseCostList
        self.t_rate_list = t_rate_list
        self.failToRecourse_old = failToRecourse_old
        self.failToRecourse_new = failToRecourse_new
        self.avg_score_list = avg_score_list
        self.balanced_acc_list = balanced_acc_list

    def save_to_csv(self, recourse_num, threshold, acceptance_rate, cost_weight, dataset, current_time, directory = ''):
        filename = f"{recourse_num}_{threshold}_{acceptance_rate}_{cost_weight}_{dataset}_{current_time}.csv"
        if directory:
            directory = directory.rstrip('/') + '/'
            filename = directory + filename


        self.avgRecourseCost.insert(0, 0)
        self.avgNewRecourseCost.insert(0, 0)
        self.avgOriginalRecourseCost.insert(0, 0)

        print(len(self.failToRecourse))
        print(len(self.avgRecourseCost))
        print(len(self.avgNewRecourseCost))
        print(len(self.avgOriginalRecourseCost))
        print(len(self.t_rate_list))
        print(len(self.failToRecourse_old))
        print(len(self.failToRecourse_new))
        print(len(self.avg_score_list))
        print(len(self.balanced_acc_list))

        # since short term balanced accuracy cannot calculate the first 2 element so insert two 0s here
        
        data = {
            'failToRecourse': self.failToRecourse,
            'avgRecourseCost': self.avgRecourseCost,
            'avgNewRecourseCost': self.avgNewRecourseCost,
            'avgOriginalRecourseCost': self.avgOriginalRecourseCost,
            't_rate': self.t_rate_list,
            'failToRecourse_old': self.failToRecourse_old,
            'failToRecourse_new': self.failToRecourse_new,
            'avg_score': self.avg_score_list,
            'balanced_acc': self.balanced_acc_list,
        }
        for key in data.keys():
            print(f"{key}: {len(data[key])}")
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        print(f"File saved as: {filename}")