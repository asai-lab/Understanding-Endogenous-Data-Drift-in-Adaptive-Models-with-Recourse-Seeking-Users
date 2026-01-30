from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.colors import ListedColormap
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.contour import QuadContourSet
from matplotlib.patches import Rectangle
import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
from Dataset.makeDataset import Dataset as makeDataset
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# important! need to change imported config according to model
# choose from Config.config, Config.MLP_config, Config.continual_config, Config.continual_MLP_config
from Config.continual_MLP_config import test, train, sample
from Models.synapticIntelligence import SynapticIntelligence

pca = PCA(2).fit(train.x)

class Helper:
    palette = sns.color_palette('muted', 2)
    cmap = ListedColormap(palette)

    def __init__(self, model: nn.Module, pca: PCA, train: Dataset, test: Dataset, sample: Dataset):
        self.model = model
        print(self.model)
        self.pca = pca
        self.train = train
        self.last_train = None
        self.test = test
        self.recoursedFail = []
        self.recoursedSuccess = []
        self.sample = sample
        self.avgRecourseCost_list = []
        self.ratioOfDifferentLabel = []
        self.fairRatio = []
        self.q3RecourseCost = []
        self.PDt = []
        self.round = 0
        self.failToRecourseOnModel = []
        self.failToRecourseOnLabel = []
        self.failToRecourse = []
        self.failToRecourseBeforeModelUpdate = []
        self.recourseModelLossList = []
        self.RegreesionModelLossList = []
        self.RegreesionModel_valLossList = []

        self.validation_list = []
        self.Ajj_performance_list = []
        self.overall_acc_list = []
        self.memory_stability_list = []
        self.memory_plasticity_list = []
        self.Aj_tide_list = []
        self.jsd_list = []

        self._hist_last: list[BarContainer]
        self._bins_last: NDArray
        self._hist_current: list[BarContainer]
        self._bins_current: NDArray
        self._sc_train: PathCollection
        self._sc_test: PathCollection
        self._sc_recourse_fail: PathCollection
        self._sc_recourse_success: PathCollection
        self._ct_test: QuadContourSet
        self._ct_train: QuadContourSet
        self.lr = 0.1
        self.si: SynapticIntelligence
        self.save_directory = None

        self.testacc = []
        self.cnt = 0
        self.avgNewRecourseCostList = []
        self.avgOriginalRecourseCostList = []
        self.historyTrainList = []
        self.train_size = 0
        self.t_rate_list = []
        self.model_params = None
        self.first_model_params = None
        self.model_shift_distance_list = []
        self.failToRecourse_old = []
        self.failToRecourse_new = [] 
        self.showRecoursedPoints = True
        self.low_cost_model_shift_list = []
        self.high_cost_model_shift_list = []
        self.low_cost_feature_ranking = []
        self.important_feature_ranking = []
        self.entropy_list = []
        self.avg_score_list = []
        self.historyTrainList_withoutRecourse = []
        self.overall_acc_list_withoutRecourse = []
        self.avg_score_on_last_train = []
        self.historyTestList = []
        self.balanced_acc_list = []

    # def draw_proba_hist(self, ax: Axes | None = None, *, label: bool = False):
    def draw_proba_hist(self, ax0: Axes = None, ax1: Axes = None, *, label: bool = False):
        if ax0 is None:
            fig0, ax0 = plt.subplots(figsize=(4, 4))
        else:
            fig0 = ax0.get_figure()
        if ax1 is None:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
        else:
            fig1 = ax1.get_figure()

        # For ax0, using self.last_train
        if hasattr(self, 'last_train') and self.last_train is not None:
            x_last = self.last_train.x
            y_last = self.last_train.y
        else:
            x_last = self.train.x
            y_last = self.train.y

        n_last = x_last.shape[0] #size of last_train data
        m_last = n_last - pt.count_nonzero(y_last) #non zero size of last_train data 
        
        with pt.no_grad():
            y_prob_last: pt.Tensor = self.model(x_last)

        # Sorts the probabilities based on the order of the labels (groups by class)
        y_last = y_last.flatten()
        y_prob_last = y_prob_last.flatten()
        y_prob_last = y_prob_last[y_last.argsort()] 

        # Creates weights for the histogram - each sample contributes a percentage to make totals sum to 100%
        w_last = np.broadcast_to(100 / n_last, n_last)

        _, self._bins_last, self._hist_last = ax0.hist(
            (y_prob_last[:m_last], y_prob_last[m_last:]),
            10,
            (0, 1),
            weights=(w_last[:m_last], w_last[m_last:]),
            rwidth=1,
            color=self.palette,
            label=(0, 1),
            ec='w',
            alpha=0.9,
        )

        ax0.legend(loc='upper center', title='Topk label')
        ax0.set_xlabel('predicted probability')
        ax0.set_ylabel('percentage')
        ax0.set_title('User Responded Dataset on Model t - 1')

        if label:
            for c in self._hist_last:
                height = map(Rectangle.get_height, c.patches)
                ax0.bar_label(
                    c,
                    [f'{h}%' if h else '' for h in height],
                    fontsize='xx-small'
                )
        ax0.set_ylim(0, 80)

        # For ax1, using self.train
        x = self.train.x
        y = self.train.y

        n = x.shape[0]
        m = n - pt.count_nonzero(y)

        with pt.no_grad():
            y_prob: pt.Tensor = self.model(x)

        y = y.flatten()
        y_prob = y_prob.flatten()
        y_prob = y_prob[y.argsort()]

        w = np.broadcast_to(100 / n, n)

        _, self._bins_current, self._hist_current = ax1.hist(
            (y_prob[:m], y_prob[m:]),
            10,
            (0, 1),
            weights=(w[:m], w[m:]),
            rwidth=1,
            color=self.palette,
            label=(0, 1),
            ec='w',
            alpha=0.9,
        )

        ax1.legend(loc='upper center', title='Topk label')
        ax1.set_xlabel('predicted probability')
        ax1.set_title('User Responded Dataset on Model t')

        if label:
            for c in self._hist_current:
                height = map(Rectangle.get_height, c.patches)
                ax1.bar_label(
                    c,
                    [f'{h}%' if h else '' for h in height],
                    fontsize='xx-small'
                )
        ax1.set_ylim(0, 80)
        
        return (fig0, ax0), (fig1, ax1)
    

    #calculate js divergence using training data after pca
    def js_divergence(self, pcaData, labelData):
        data1 = pcaData[labelData.flatten() == 0]
        data2 = pcaData[labelData.flatten() == 1]

        # Fit KDEs
        kde1 = KernelDensity(kernel='gaussian', bandwidth=1).fit(data1)
        kde2 = KernelDensity(kernel='gaussian', bandwidth=1).fit(data2)

        # Create a grid that covers data 
        combined_data = np.vstack([data1, data2])
        x_min, y_min = combined_data.min(axis=0) - 1 # Add a margin
        x_max, y_max = combined_data.max(axis=0) + 1 # Add a margin
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Evaluate KDEs on the new grid
        log_dens1 = kde1.score_samples(grid_points)
        log_dens2 = kde2.score_samples(grid_points)
        dens1 = np.exp(log_dens1)
        dens2 = np.exp(log_dens2)

        # Normalize the densities to sum to 1
        dens1 /= dens1.sum()
        dens2 /= dens2.sum()

        js_divergence_score = jensenshannon(dens1, dens2)
        self.jsd_list.append(js_divergence_score)


    # def draw_dataset_scatter(self, axes: tuple[Axes, Axes] | None = None):
    def draw_dataset_scatter(self, axes: [Axes, Axes] = None):
        if axes is None:
            # Increase the figure size for larger plotting area
            fig, (ax0, ax1) = plt.subplots(
                1, 2,
                sharex=True,
                sharey=True,
                figsize=(8, 4),
                layout='constrained'
            )
        else:
            ax0, ax1 = axes
            fig = ax0.get_figure()

        prop = dict(cmap=self.cmap, s=40, vmin=0., vmax=1., lw=0.8, ec='w')
        
        # Standard scatter plot for training data
        self._sc_train = ax0.scatter(
            *pca.transform(self.train.x).T,
            c=self.train.y,
            **prop
        )
        
        if(self.showRecoursedPoints):
            # Extract the recoursedFail points from training data
            if hasattr(self, 'recoursedFail') and len(self.recoursedFail) > 0:
                # Convert recoursedFail to tensor indices if it's a list
                if isinstance(self.recoursedFail, list):
                    recourse_fail_indices = pt.tensor(self.recoursedFail, dtype=pt.long)
                else:
                    recourse_fail_indices = self.recoursedFail
                    
                # Get the PCA-transformed coordinates of recourse points
                recourse_fail_points = pca.transform(self.train.x[recourse_fail_indices])
                
                # Plot the recoursedFail points in green over the original scatter plot
                self._sc_recourse_fail = ax0.scatter(
                    *recourse_fail_points.T,
                    s=40,  
                    lw=0.8, 
                    ec='w',
                    color='purple',      
                )
                
            else:
                # Initialize with empty data so the attribute exists
                self._sc_recourse_fail = ax0.scatter(
                    [], 
                    [], 
                    s=40,  
                    lw=0.8, 
                    ec='w',
                    color='purple',
                )
            
            # Extract the recoursedSuccess points from training data
            if hasattr(self, 'recoursedSuccess') and len(self.recoursedSuccess) > 0:
                # Convert recoursedSuccess to tensor indices if it's a list
                if isinstance(self.recoursedSuccess, list):
                    recourse_success_indices = pt.tensor(self.recoursedSuccess, dtype=pt.long)
                else:
                    recourse_success_indices = self.recoursedSuccess
                    
                # Get the PCA-transformed coordinates of recoursedSuccess points
                recourse_success_points = pca.transform(self.train.x[recourse_success_indices])
                
                # Plot the recoursedFail points in green over the original scatter plot
                self._sc_recourse_success = ax0.scatter(
                    *recourse_success_points.T,
                    s=40,  
                    lw=0.8, 
                    ec='w',
                    color='red',      
                )
                
            else:
                # Initialize with empty data so the attribute exists
                self._sc_recourse_success = ax0.scatter(
                    [], 
                    [], 
                    s=40,  
                    lw=0.8, 
                    ec='w',
                    color='red',
                )

        if(self.showRecoursedPoints):
            handles = []
            labels = []
            # Add training scatter points to legend (if needed)
            if hasattr(self, '_sc_train'):
                train_handles, train_labels = self._sc_train.legend_elements()
                handles.extend(train_handles)
                labels.extend([label for label in train_labels])

            # Add recourse success points to legend
            if hasattr(self, '_sc_recourse_success'):
                handles.append(self._sc_recourse_success)
                labels.append('R_1')

            # Add recourse fail points to legend
            if hasattr(self, '_sc_recourse_fail'):
                handles.append(self._sc_recourse_fail)
                labels.append('R_0')

            # Create the combined legend
            ax0.legend(handles, labels, loc='upper right', title='topk label')
        
        else:
            ax0.legend(
                *self._sc_train.legend_elements(),
                loc='upper right',
                title='topk label'
            )

        # Add PCA variance to labels
        ax0.set_xlabel(f'PCA1')
        ax0.set_ylabel(f'PCA2')
        ax0.grid(alpha=0.75)
        ax0.set_title('User Responded Dataset at time t')
        
        with pt.no_grad():
            y_prob: pt.Tensor = self.model(test.x)

        y_prob = y_prob.flatten()
        y_pred = y_prob.greater(0.5)

        self._sc_test = ax1.scatter(
            *pca.transform(test.x).T,
            c=y_pred,
            **prop
        )
        ax1.legend(
            *self._sc_test.legend_elements(),
            loc='upper right',
            title='model label'
        )

        x0, x1 = ax0.get_xlim()
        y0, y1 = ax0.get_ylim()
        
        # Increase the expand factor for larger PCA area
        expand_factor = 1.0
        x_range = (x1 - x0) * expand_factor
        y_range = (y1 - y0) * expand_factor

        # Set new limits
        ax0.set_xlim([x0 - x_range, x1 + x_range])
        ax0.set_ylim([y0 - y_range, y1 + y_range])
        ax1.set_xlim([x0 - x_range, x1 + x_range])
        ax1.set_ylim([y0 - y_range, y1 + y_range])
        
        # Increase grid resolution for smoother contours
        n = 32
        x_expanded = np.linspace(x0 - x_range, x1 + x_range, n)
        y_expanded = np.linspace(y0 - y_range, y1 + y_range, n)

        xy = np.meshgrid(x_expanded, y_expanded)

        z = pca.inverse_transform(np.c_[xy[0].ravel(), xy[1].ravel()])
        z = pt.tensor(z, dtype=pt.float)

        # Evaluate the model on the expanded grid
        with pt.no_grad():
            z = self.model(z)
        z = z.view(n, n)
        
        self._ct_test = ax1.contourf(
            *xy, z, 10,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            alpha=0.9,
            zorder=0,
        )
        
        # Also add the contour to the first plot
        self._ct_train = ax0.contourf(
            *xy, z, 10,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            alpha=0.9,
            zorder=0,
        )
        
        fig.colorbar(self._ct_test, ax=ax1, label='probability')
        ax1.grid(alpha=0.75) 
        
        ax1.set_xlabel(f'PCA1')
        ax1.set_title('Initial Distribution Dataset')

        return fig, axes

    def draw_all(self):
        sf: list[SubFigure]
        fig = plt.figure(figsize=(8, 8), layout='constrained')
        sf = fig.subfigures(2, 2)
        ax0 = sf[0, 0].subplots()
        ax1 = sf[0, 1].subplots()
        bottom_sf = fig.subfigures(2, 1)[1]
        ax2, ax3 = bottom_sf.subplots(1, 2, sharex=True, sharey=True)
        self.draw_proba_hist(ax0, ax1)
        self.draw_dataset_scatter((ax2, ax3))
        return fig, (ax0, ax1, ax2, ax3)

    def animate_all(self, frames: int = 120, fps: int = 10, *, inplace: bool = False):
        fig, (ax0, ax1, ax2, ax3) = self.draw_all()

        def init():
            return *ax0.patches, *ax1.patches, *ax2.collections, *ax3.collections


        def func(frame):
            """
            For each frame, it does four main things: 
            takes a "before" snapshot, 
            updates the state of the simulation, 
            draws the "after" snapshot, 
            and updates the boundary in the plots.
            """

            # PCA for training data
            pca_train = PCA(2).fit(train.x)

            # Separate PCA for test data
            pca_test = PCA(2).fit(test.x)

            fig.suptitle(f't = {frame}', ha='left', x=0.01, size='small')

            # Update histograms for last_train (ax0)
            # store the last training data here before update the train data in self.update()
            self.last_train = makeDataset(self.train.x.clone(), self.train.y.clone())
            # compute the output probability of last training data on last model before update
            with pt.no_grad():
                if(self.last_train is not None):
                    y_prob_last: pt.Tensor = self.model(self.last_train.x)
                    y_last = self.last_train.y.flatten()
                    n_last = self.last_train.x.shape[0]
                else:
                    y_prob_last = None

            self.update(self.model, self.train, self.sample, self.recoursedFail, self.recoursedSuccess)

            # Update histograms for last_train (ax0)
            # here we handle the case that in the initial rounds there is no recourse,
            # so there is no modification on train data and model. We set last train data = current train data
            with pt.no_grad():
                if(self.last_train is None):
                    y_prob_last: pt.Tensor = self.model(self.train.x)
                    y_last = self.train.y.flatten()
                    n_last = self.train.x.shape[0]     

            y_prob_last = y_prob_last.flatten()
            m_last = n_last - pt.count_nonzero(y_last)
            rank_last = y_prob_last[y_last.argsort()]

            # Update histogram for last train data (ax0)
            for b, r in zip(self._hist_last, (rank_last[:m_last], rank_last[m_last:])):
                height, _ = np.histogram(r, self._bins_last, range=(0, 1))
                for rect, h in zip(b.patches, height * (100 / n_last)):
                    rect.set_height(h)

            # Update histograms for current train (ax1)
            with pt.no_grad():
                y_prob_current: pt.Tensor = self.model(self.train.x)

            y_current = self.train.y.flatten()
            y_prob_current = y_prob_current.flatten()
            n_current = self.train.x.shape[0]
            m_current = n_current - pt.count_nonzero(y_current)
            rank_current = y_prob_current[y_current.argsort()]

            # Update histogram for current train data (ax1)
            for b, r in zip(self._hist_current, (rank_current[:m_current], rank_current[m_current:])):
                height, _ = np.histogram(r, self._bins_current, range=(0, 1))
                for rect, h in zip(b.patches, height * (100 / n_current)):
                    rect.set_height(h)

            # Use pca_train for training data
            self._sc_train.set_offsets(pca_train.transform(train.x))
            self._sc_train.set_array(train.y.flatten())

            if(self.showRecoursedPoints):
                if hasattr(self, 'recoursedFail') and len(self.recoursedFail) > 0:
                    # Get the indexed array
                    indexed_array = self.train.x[self.recoursedFail]
                    
                    # Check if the indexed array actually has samples
                    if indexed_array.shape[0] > 0:
                        self._sc_recourse_fail.set_offsets(pca_train.transform(indexed_array))
                        
                if hasattr(self, 'recoursedSuccess') and len(self.recoursedSuccess) > 0:
                    # Get the indexed array
                    indexed_array = self.train.x[self.recoursedSuccess]
                    
                    # Check if the indexed array actually has samples
                    if indexed_array.shape[0] > 0:
                        self._sc_recourse_success.set_offsets(pca_train.transform(indexed_array))

            # calculate js divergence of pca training data
            self.js_divergence(self._sc_train.get_offsets(), self._sc_train.get_array())
            with pt.no_grad():
                y_prob: pt.Tensor = self.model(test.x)

            y_prob = y_prob.flatten()
            y_pred = y_prob.greater(0.5)

            # Use pca_test for test data
            self._sc_test.set_offsets(pca_test.transform(test.x))
            self._sc_test.set_array(y_pred)


            ax2.relim()
            ax2.autoscale_view()

            ax3.relim()
            ax3.autoscale_view()

            for c in self._ct_test.collections:
                c.remove()

            for c in self._ct_train.collections:
                c.remove()

            # For training plot (ax2)
            x0_train, x1_train = ax2.get_xlim()
            y0_train, y1_train = ax2.get_ylim()
            n = 32
            xy_train = np.mgrid[x0_train: x1_train: n * 1j, y0_train: y1_train: n * 1j]
            z_train = pca_train.inverse_transform(xy_train.reshape(2, n * n).T)
            z_train = pt.tensor(z_train, dtype=pt.float)
            with pt.no_grad():
                z_train: pt.Tensor = self.model(z_train)
            z_train = z_train.view(n, n)

            self._ct_train: QuadContourSet = ax2.contourf(
                *xy_train, z_train, 10,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                alpha=0.9,
                zorder=0,
            )

            # For test plot (ax3)
            x0_test, x1_test = ax3.get_xlim()
            y0_test, y1_test = ax3.get_ylim()
            xy_test = np.mgrid[x0_test: x1_test: n * 1j, y0_test: y1_test: n * 1j]
            z_test = pca_test.inverse_transform(xy_test.reshape(2, n * n).T)
            z_test = pt.tensor(z_test, dtype=pt.float)
            with pt.no_grad():
                z_test: pt.Tensor = self.model(z_test)
            z_test = z_test.view(n, n)

            self._ct_test: QuadContourSet = ax3.contourf(
                *xy_test, z_test, 10,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                alpha=0.9,
                zorder=0,
            )
            save_interval = 10
            
            n_recourse_success = len(self.recoursedSuccess) if hasattr(self, 'recoursedSuccess') else 0
            n_recourse_fail = len(self.recoursedFail) if hasattr(self, 'recoursedFail') else 0
            sc_train_labels = self._sc_train.get_array()  # this holds labels for plotted train points

            # n_label_1 = np.sum(sc_train_labels == 1)
            # n_label_0 = np.sum(sc_train_labels == 0)
            # print("n_label_1: ",n_label_1)
            # print("n_label_0: ",n_label_0)
            # print("n_recourse_success: ",n_recourse_success)
            # print("n_recourse_fail: ",n_recourse_fail)

            return *ax0.patches, *ax1.patches, *ax2.collections, *ax3.collections

        return FuncAnimation(
            fig, func, frames, init,
            interval=1000 // fps,
            repeat=False,
            blit=True,
            cache_frame_data=False
        )

    def calculate_balanced_accuracy(self, predicted_results, actual_labels, threshold=0.5):
        y_pred = (predicted_results >= threshold).float()
        y_true_np = actual_labels.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        bal_acc = balanced_accuracy_score(y_true_np, y_pred_np)
        return bal_acc

    def _weight_func(self, x):
        # can set different weight function to adjust the weight of historical data
        return 1
    
    def _get_weight(self, observe_range):
        weights = []
        for i in range (1, observe_range + 1):
            weights.append(self._weight_func(i))

        sum = 0
        for i in weights:
            sum += i

        return [w / sum for w in weights]

    def calculate_STBA(self, kth_model: nn.Module, jth_data_after_recourse: list, rangenum, option = 'balanced'):
        # if the round of past data is not enough, then just use all the past data
        rangenum = min(rangenum, len(jth_data_after_recourse))
        # skip the current round data, only evaluate on the past data
        rangenum -= 1

        if jth_data_after_recourse:
            kth_model.eval()
            sum = 0
            weights = self._get_weight(rangenum)
            
            # do each historical task
            for j in range(-2, -rangenum - 2, -1):
                pred = kth_model(jth_data_after_recourse[j].x)
                acc = self.calculate_balanced_accuracy(pred, jth_data_after_recourse[j].y) * weights[j+1]
                sum += acc
            return sum

        print("jth_data_after_recourse cannot be empty")
        return None