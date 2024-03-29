import torch
import torch.nn.functional as F
import torch.nn as nn

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        prediction = F.log_softmax(input, dim=-1)
        loss = F.nll_loss(prediction, target)
        return loss


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.linear = torch.nn.Sequential(torch.nn.Linear(3 * 64 * 64, 100),
                                          torch.nn.Linear(100, 6))

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x_flat = x.view(x.shape[0], -1)
        return self.linear(x_flat)
    
    def predict(self, image):
      return torch.nn.Softmax(dim=1)(self(image)).argmax(1)


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.linear = nn.Sequential(nn.Linear(3 * 64 * 64, 95), nn.ReLU(), 
          nn.Linear(95, 6))

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x_flat = x.view(x.shape[0], -1)
        return self.linear(x_flat)
    
    def predict(self, image):
      return torch.nn.Softmax(dim=1)(self(image)).argmax(1)


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
