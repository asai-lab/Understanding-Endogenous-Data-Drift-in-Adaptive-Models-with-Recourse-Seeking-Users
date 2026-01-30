import torch as pt
from torch import nn
from copy import deepcopy
import math
import os
import sys
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Experiment_Helper.helper import Helper, pca
from Experiment_Helper.auxiliary import getWeights, update_train_data, FileSaver
from Models.synapticIntelligence import continual_training
from Models.recourse import recourse
from Config.continual_MLP_config import train, test, sample, si, dataset, POSITIVE_RATIO # modified parameters for observations
from Dataset.makeDataset import Dataset

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_file_name = os.path.basename(current_file_path)
current_file_name = os.path.splitext(current_file_name)[0]

DIRECTORY = os.path.join(current_directory, f"{current_file_name}_output")

# modified parameters for observations
THRESHOLD = 0.7         #0.5 0.7 0.9
RECOURSENUM = 0.5       #0.2 0.5 0.7
COSTWEIGHT = 'uniform'  #uniform log inverse_gamma
DATASET = dataset

try:
    os.makedirs(DIRECTORY, exist_ok=True)
    print(f"Folder '{DIRECTORY}' is ready.")
except Exception as e:
    print(f"An error occurred: {e}")

class Exp3(Helper):
    '''
    1. perform recourse on dataset D
    2. labeling D with topk method
    3. continual training the model with the updated dataset
    4. calculate metrics
    '''

    def update(self, model: nn.Module, train: Dataset, sample: Dataset, recoursedFail, recoursedSuccess):
        print("round: ",self.round)
        self.round += 1

        #save model parameters
        self.model_params = deepcopy(self.model.state_dict())

        if self.round != 1:
            #randomly select from self.sample with size of train and label it with model
            self.train, isNewList = update_train_data(self.train, self.sample, self.model, 'mixed')

            # find training data with label 0 and select 1/5 of them
            data, labels = self.train.x, self.train.y
            label_0_indices = pt.where(labels == 0)[0]
            shuffled_indices = pt.randperm(len(label_0_indices))
            label_0_indices = label_0_indices[shuffled_indices]
            num_samples = math.floor(len(label_0_indices) * RECOURSENUM)
            selected_indices = label_0_indices[:num_samples]

            # perform recourse on the selected subset
            selected_subset = Dataset(data[selected_indices], labels[selected_indices].unsqueeze(1))
            recourse_weight = getWeights(self.train.x.shape[1], COSTWEIGHT)
            recoursed, action = recourse(
                self.model,
                selected_subset,
                100,
                recourse_weight,
                loss_list=[],
                threshold=THRESHOLD,
                cost_list=self.avgRecourseCost_list,
                q3RecourseCost=self.q3RecourseCost,
                recourseModelLossList=self.recourseModelLossList,
                isNew = pt.tensor([]) if isNewList.numel() == 0 else isNewList[selected_indices],
                new_cost_list=self.avgNewRecourseCostList,
                original_cost_list=self.avgOriginalRecourseCostList
            )
            self.train.x[selected_indices] = recoursed.x

            # update the labels of D using topk method
            with pt.no_grad():
                y_prob_all: pt.Tensor = self.model(self.train.x)
            sorted_indices = pt.argsort(y_prob_all[:, 0], dim=0, descending=True)
            cutoff_index = int(len(sorted_indices) * POSITIVE_RATIO)
            mask = pt.zeros_like(y_prob_all)
            mask[sorted_indices[:cutoff_index]] = 1
            self.train.y = mask.float().squeeze(1)


            # train the model with the updated dataset
            continual_training(self.si, self.train, 50, tao = 0.00001, observe_range=7)

            
        #calculate metrics: ========================================================================
        
        #calculate short term accuracy
        current_data = Dataset(self.train.x, self.train.y)
        self.historyTrainList.append(current_data)
        with pt.no_grad():
            y_prob_test: pt.Tensor = self.model(self.test.x)
        y_prob_test = y_prob_test.squeeze(1)
        y_pred_test = (y_prob_test > 0.5).float()
        self.test.y = y_pred_test
        current_test = Dataset(self.test.x, self.test.y)
        self.historyTestList.append(current_test)
        self.balanced_acc_list.append(self.calculate_STBA(self.model, self.historyTestList, 7, "balanced"))

        if self.round != 1: #skip the first round to save the initial data in csv
            #calculate ftr
            fail_positions = pt.where(self.train.y[selected_indices] == 0)[0]
            success_positions = pt.where(self.train.y[selected_indices] == 1)[0]
            self.recoursedFail = selected_indices[fail_positions]
            self.recoursedSuccess = selected_indices[success_positions]
            recourseFailCnt = fail_positions.shape[0] if fail_positions.shape[0] > 0 else 0
            recourseFailRate = recourseFailCnt / len(self.train.y[selected_indices])
            self.failToRecourse.append(recourseFailRate)

            if isNewList.numel() != 0:
                new_indices = isNewList[selected_indices]

                #calculate ftr_old
                old_selected_indices = selected_indices[new_indices == False]
                recourseFailCnt_old = pt.where(self.train.y[old_selected_indices] == 0)[0].shape[0]
                recourseFailRate_old = recourseFailCnt_old / len(self.train.y[old_selected_indices])
                self.failToRecourse_old.append(recourseFailRate_old)
                
                #calculate ftr_new
                new_selected_indices = selected_indices[new_indices == True]
                recourseFailCnt_new = pt.where(self.train.y[new_selected_indices] == 0)[0].shape[0]
                recourseFailRate_new = recourseFailCnt_new / len(self.train.y[new_selected_indices])
                self.failToRecourse_new.append(recourseFailRate_new)
                
            else:
                self.failToRecourse_old.append(0)
                self.failToRecourse_new.append(0)

        else:
            self.failToRecourse.append(0)
            self.failToRecourse_old.append(0)
            self.failToRecourse_new.append(0)

        #calculate t_rate
        with pt.no_grad():
            y_prob: pt.Tensor = self.model(test.x)
        #calculate the ratio of 1s and 0s in the test data
        num_ones = pt.where(y_prob > 0.5)[0].shape[0]
        num_zeros = len(y_prob) - num_ones
        t_rate = num_ones / num_zeros
        self.t_rate_list.append(t_rate)
        

        # calculate average score of the model before sigmoid
        with pt.no_grad():
            x = self.test.x
            x = pt.relu(self.model.layers[0](x))      # First Linear + ReLU
            x = pt.relu(self.model.layers[2](x))      # Second Linear + ReLU
            score = self.model.layers[4](x)              # Final Linear (before Sigmoid)
            avg_score = score.mean()
            self.avg_score_list.append(avg_score.item())

        print("====================================================")



exp3 = Exp3(si.model, pca, train, test, sample)
exp3.si = si
exp3.save_directory = DIRECTORY
current_time = datetime.datetime.now().strftime("%d-%H-%M")
ani1 = exp3.animate_all(101) #100 rounds (first round do nothing, just to save initial state)
ani1.save(os.path.join(DIRECTORY, f"{RECOURSENUM}_{THRESHOLD}_{POSITIVE_RATIO}_{COSTWEIGHT}_{DATASET}_{current_time}.gif"))
# in the generated gif, each round PCA is not continuous, because PCA is fitted on the training data of that round

# save to csv
FileSaver(exp3.failToRecourse,  
          exp3.avgRecourseCost_list, 
          exp3.avgNewRecourseCostList,
          exp3.avgOriginalRecourseCostList,
          exp3.t_rate_list,
          exp3.failToRecourse_old,
          exp3.failToRecourse_new,
          exp3.avg_score_list,
          exp3.balanced_acc_list
        ).save_to_csv(RECOURSENUM, THRESHOLD, POSITIVE_RATIO, COSTWEIGHT, DATASET, current_time, DIRECTORY)