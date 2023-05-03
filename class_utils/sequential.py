from typing import Union
import numpy as np

try:
    import torch
    from torch import Tensor
    _torch_available = True
except ImportError:
    class Tensor: pass
    _torch_available = False

def sliding_window(
    seq: Union[np.ndarray, Tensor],
    win_size: int,
    step_size: int = None,
    writeable: bool = False,
    batch_first: bool = True
) -> Union[np.ndarray, Tensor]:
    """Splits a sequence into windows of a given size and with
    the specified overlap.

    Args:
        seq (Union[np.ndarray, Tensor]): An array/tensor with dimensions
            (seq, feature dims), i.e. with the sequence dimension first
            and any other dimensions next.
        win_size (int): The number of samples in the seq dimension per window. 
        step_size (int, optional): The number of samples in the seq dimensions
            to move after extracting a window. I.e. if step_size == win_size
            there is no overlap between adjacent windows. If None, defaults
            to win_size.
        writeable (bool, optional): An argument passed to numpy's as_strided
            function: if you are going to modify the returned array, the data
            needs to be copied and you need to set writeable to True.
            Defaults to False.
        batch_first (bool, optional): Determines the shape of the returned
            array. If False, the shape is (win_size, # of windows, feature dims).
            If True, the shape is (# of windows, win_size, feature dims).

    Returns:
        numpy array: Returns an array with windows extracted from the input seq.
            The shape of the returned array depends on batch_first.
    """
    if step_size is None: step_size = win_size

    if isinstance(seq, Tensor):
        res = seq.unfold(0, win_size, step_size).swapaxes(1, 2)
    else:
        seq_orig = seq
        win_size_orig = win_size
        
        if len(seq_orig.shape) > 1:
            seq = seq.reshape(-1)
            win_size *= np.product(seq_orig.shape[1:])
            step_size *= np.product(seq_orig.shape[1:])
            
        stride_inc = (step_size - 1) * seq.dtype.itemsize
        stride, = seq.strides
    
        res = np.lib.stride_tricks.as_strided(seq,
                  [int((seq.shape[0] - win_size) / step_size + 1), win_size],
                  strides = (stride + stride_inc, stride), writeable=writeable)
    
        if len(seq_orig.shape) > 1:
            res = res.reshape((res.shape[0], win_size_orig) + (seq_orig.shape[1:]))

    if not batch_first:
        res = res.swapaxes(0, 1)
    
    return res

if _torch_available:
    from numbers import Number
    import logging

    class RandomWindowDataset(torch.utils.data.IterableDataset):
        def __init__(self,
            signal: Union[np.ndarray, Tensor],
            win_size: int,
            prefix_size: int = 0,
            autoregressive: bool = True,
            auto_featuredim: bool = True
        ):
            """
            A dataset class that takes in a signal and produces batches of
            randomly sampled windows from it.

            Args:
                signal (Union[np.ndarray, Tensor]): 
            """
            super().__init__()
            assert len(signal) > win_size
            assert prefix_size < win_size
            
            self.signal = signal
            self.win_size = win_size
            self.prefix_size = prefix_size
            self.autoregressive = autoregressive
            self.auto_featuredim = auto_featuredim
        
        def __next__(self):
            chan_ind = torch.randint(0, self.signal.shape[1], tuple())
            win_ind = torch.randint(0, self.signal.shape[0]-self.win_size, tuple())
            x = self.signal[win_ind:win_ind+self.win_size, chan_ind, ...]

            if self.auto_featuredim and len(x.shape) == 1:
                x = x.reshape(-1, 1)

            if self.autoregressive:
                return x[:-1], x[1+self.prefix_size:]
            else:
                return x

        def __iter__(self):
            return self

    class SlidingWindowDataset(torch.utils.data.Dataset):
        def __init__(self,
            signal, win_size, step_size=None, prefix_size=0,
            autoregressive=True, auto_featuredim=True
        ):
            super().__init__()
            assert len(signal) > win_size
            assert prefix_size < win_size

            self.prefix_size = prefix_size
            self.autoregressive = autoregressive
            self.auto_featuredim = auto_featuredim

            self.wins = sliding_window(signal, win_size, step_size=step_size)
            self.wins = self.wins.moveaxis(-1, 0).flatten(0, 1)

        def __getitem__(self, isample):
            x = self.wins[isample]

            if self.auto_featuredim and len(x.shape) == 1:
                x = x.reshape(-1, 1)

            if self.autoregressive:
                return x[:-1], x[1+self.prefix_size:]
            else:
                return x

        def __len__(self):
            return self.wins.shape[0]
