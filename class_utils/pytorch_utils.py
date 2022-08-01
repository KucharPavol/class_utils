import torch

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
