import torch
from collections import deque

class EarlyStopping:
    """
    A convenience class that implements a simple early stopping mechanism.

    Arguments:
        patience: The number of epochs to wait before terminating the training.
        checkpoint_path: The path to the file where the model state is saved.
            If None, no checkpoint will be saved.
    """
    def __init__(self, patience=10, checkpoint_path="best_model.pt"):
        self.patience = patience
        self.counter = 0
        self.best_loss = torch.inf
        self.checkpoint_path = checkpoint_path

    def __call__(self, validation_loss, model=None):
        """
        Checks if the validation loss is decreasing. If not and the patience
        limit has been reached, returns True. Otherwise, returns False.

        If the validation loss has improved since the last step and a model
        is passed as an argument, it is saved to the checkpoint path.

        Arguments:
            validation_loss: The current validation loss.
            model: The model to save if the validation loss is the best so far.
        """

        if self.best_loss > validation_loss:
            self.best_loss = validation_loss
            
            if not model is None and not self.checkpoint_path is None:
                torch.save(model.state_dict(), self.checkpoint_path)

            self.counter = 0
            
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
                
        return False

class BestModelCheckpointer:
    """
    A convenience class that tracks the best validation loss and saves the
    model every time it improves.
    """
    def __init__(self, checkpoint_path="best_model.pt"):
        self.checkpoint_path = checkpoint_path
        self.best_loss = torch.inf

    def __call__(self, validation_loss, model=None):
        if self.best_loss > validation_loss:
            self.best_loss = validation_loss
            
            if not model is None and not self.checkpoint_path is None:
                torch.save(model.state_dict(), self.checkpoint_path)

def freeze_except_last(model, freeze=True, num_last=None):
    num_unfrozen = 0
    q = deque()
    q.extend(model.children())

    if num_last is None:
        if freeze: num_last = 1
        else: num_last = 0

    while len(q):
        layer = q.pop()
        num_children = 0

        if hasattr(layer, "children"):
            children = list(layer.children())
            num_children = len(children)
            q.extend(children)

        if not num_children:
            if num_unfrozen < num_last:
                params = list(layer.parameters())

                for param in params:
                    param.requires_grad = freeze

                if len(params):
                    num_unfrozen += 1
                    
            else:
                for param in layer.parameters():
                    param.requires_grad = not freeze

    return model
