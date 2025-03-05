import numpy as np
import torch
from algo import ValueFunctionWithApproximation

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self,s):
        # TODO: implement this method
        self.model.eval()
        with torch.no_grad():
            s_tensor = torch.FloatTensor(s).unsqueeze(0)
            value = self.model(s_tensor)
        return value.item()

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        self.model.train()
        s_tau_tensor = torch.tensor(s_tau, dtype=torch.float32).unsqueeze(0)
        G_tensor = torch.tensor([G], dtype=torch.float32)

        self.optimizer.zero_grad()
        value_estimate = self.model(s_tau_tensor)
        loss = self.loss_fn(value_estimate, G_tensor)
        loss.backward()
        self.optimizer.step()
        return None

