import torch


class ignore_default_device:
    """Context manager that temporarily sets the default device to CPU.
    
    When entering the context, sets the default device to CPU. When exiting,
    restores the original default device.
    """
    def __init__(self):
        self.original_device = None

    def __enter__(self):
        self.original_device = torch.get_default_device()
        torch.set_default_device('cpu')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_default_device(self.original_device)
