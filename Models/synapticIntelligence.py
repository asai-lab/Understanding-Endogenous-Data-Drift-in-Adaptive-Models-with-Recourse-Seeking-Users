import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class SynapticIntelligence:
    def __init__(self, model):
        self.model = model
        self.prev_params = {}          # For per-optimizer-step parameter change Δθ
        self.prev_task_params = {}     # θ_k — Anchor parameters from the end of the last task
        self.omega = {}                # Ω_k^t — Total parameter importance accumulated over tasks
        self.omega_list = []           # Stores per-task importance components for custom logic
        self.path_integrals = {}       # ω_k^u — Path integral accumulated during a single task
        self.epsilon = 1e-8            # ε — Stability constant to avoid division by zero
        
        for name, param in model.named_parameters():
            self.prev_params[name] = param.data.clone()
            self.prev_task_params[name] = param.data.clone()
            self.omega[name] = pt.zeros_like(param)
            self.path_integrals[name] = pt.zeros_like(param)

    def update_path_integral(self):
        """
        Call after optimizer.step(). Accumulate grad · delta_param.
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                delta_param = param.data - self.prev_params[name]
                self.path_integrals[name] += param.grad.detach() * delta_param
                self.prev_params[name] = param.data.clone()

    # can change to other weight functions
    def _weight_func(self, x):
        return 1
    
    def _get_weight(self, observe_range):
        weights = []
        for i in range (1, observe_range + 1):
            weights.append(self._weight_func(i))
        weights_tensor = pt.tensor(weights, dtype=pt.float32)
        return weights_tensor / sum(weights_tensor)

    def consolidate(self, observe_range):
        """
        At the end of a task: calculate ω^ν / Δ^ν² and sum up with weight.
        """
        current_task_omega = {}
        weights = self._get_weight(observe_range)

        for name, param in self.model.named_parameters():
            delta = param.data - self.prev_task_params[name]  # Δ^k = θ_k - θ_k-1
            denominator = delta ** 2 + self.epsilon
            omega_update = self.path_integrals[name] / denominator

            current_task_omega[name] = omega_update

            if name in self.omega:

                if len(self.omega_list) >= observe_range:
                    omega_sum = pt.zeros_like(param)
                    weights = self._get_weight(observe_range)
                    for i in range(-2, -observe_range - 1, -1):
                        omega_sum += self.omega_list[i][name] * weights[i]
                    omega_sum += omega_update * weights[-1]
                    self.omega[name] = omega_sum
                else:
                    for i in range(len(self.omega_list)):
                        self.omega[name] += self.omega_list[i][name]
                    self.omega[name] += omega_update
                
            else:
                self.omega[name] = omega_update

            
            # Save current params as "previous task parameter" for the next Δ^k
            self.prev_task_params[name] = param.data.clone()
            self.path_integrals[name].zero_()
            self.prev_params[name] = param.data.clone()

        self.omega_list.append(current_task_omega)

    def compute_si_loss(self, tao):
        """
        Computes the SI regularization loss term:
        τ * ∑_k Ω_k^t * (θ_k-1 - θ_k)²
        """
        si_loss = 0
        param_count = 0
        for name, param in self.model.named_parameters():
            if name in self.omega:
                si_loss += pt.sum(self.omega[name] * (param - self.prev_task_params[name]) ** 2)
                param_count += param.numel()
        
        # to make the regularization strength τ independent of the model's size
        if param_count > 0:
            si_loss = si_loss / param_count

        return tao * si_loss
    
    # Training loop with SI
def continual_training(si: SynapticIntelligence, dataset: Dataset, max_epochs: int, loss_list=None, tao=0.5, observe_range=5):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(si.model.parameters(), 0.1, weight_decay=0.001)

    si.model.train()
    epoch_loss = []
    
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for _ in range(max_epochs):
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = si.model(X_batch)
            outputs = outputs.squeeze()
            
            if outputs.dim() != Y_batch.dim():
                outputs = outputs.unsqueeze(-1)
            
            # Calculate loss with SI regularization
            task_loss = criterion(outputs, Y_batch)
            si_loss = si.compute_si_loss(tao)
            loss = task_loss + si_loss
            running_loss += loss.item() * X_batch.size(0)
            
            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update path integral with current gradients
            si.update_path_integral()

            
            if loss_list is not None:
                loss_list.append(loss.item())
        
        train_loss = running_loss / len(train_loader.dataset)
        epoch_loss.append(train_loss)

        if(_ == max_epochs - 1):
            print(f"Model Loss: {train_loss:.4f}")
    
    
    # After task training is complete, consolidate knowledge
    si.consolidate(observe_range)