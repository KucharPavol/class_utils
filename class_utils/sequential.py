import numpy as np

def sliding_window(seq, win_size, step_size=None, writeable=False, batch_first=False):
    """Cuts up a sequence into windows.

    Args:
        seq (numpy array): An array with dimensions (seq, feature dims), i.e.
            with the sequence dimension first and any other dimensions next.
        win_size (integer): The number of samples in the seq dimension per window. 
        step_size (integer, optional): The number of samples in the seq dimensions to
            move after extracting a window. I.e. if step_size == win_size
            there is no overlap between adjacent windows. If None, defaults
            to win_size.
        writeable (bool, optional): An argument passed to numpy's as_strided
            function: if you are going to modify the returned array, the data
            needs to be copied and you need to set writeable to True.
            Defaults to False.
        batch_first (bool, optional): Determines the shape of the returned
            array. If False (default) the shape is (win_size, # of windows, feature dims).
            If True, the shape is (# of windows, win_size, feature dims).

    Returns:
        numpy array: Returns an array with windows extracted from the input seq.
            The shape of the returned array depends on batch_first.
    """    

    seq_orig = seq
    if step_size is None: step_size = win_size
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