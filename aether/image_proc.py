from typing import Union, Optional

import torch
import numpy as np
from PIL import Image
import aether.maths as maths

class ImageNormalizer:
    """Normalize or unnormalize an image.
    """
    def __init__(self, min_quantile: float = 0.01, max_quantile: float = 0.99):
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.min_val = None
        self.max_val = None
        
    def normalize(self, img: torch.Tensor):
        """Normalize an image to [0, 1].
        
        Parameters
        ----------
        img: torch.Tensor
            The image to normalize.
            
        Returns
        -------
        torch.Tensor
            The normalized image.
        """
        min_val = torch.quantile(img, self.min_quantile)
        max_val = torch.quantile(img, self.max_quantile)
        self.min_val = min_val
        self.max_val = max_val
        return ((img - min_val) / (max_val - min_val)).clip(0, 1)
    
    def unnormalize(self, img: torch.Tensor):
        """Unnormalize an image from [0, 1].
        
        Parameters
        ----------
        img: torch.Tensor
            The image to unnormalize.
            
        Returns
        -------
        torch.Tensor
            The unnormalized image.
        """
        return img * (self.max_val - self.min_val) + self.min_val


def match_mean_std(
    img: torch.Tensor, 
    reference: torch.Tensor, 
    roi_slicer: Optional[tuple[slice, ...]] = None,
    mask: Optional[torch.Tensor] = None,
):
    """Match the mean and std of the image to the target mean and std within the ROI.
    """
    if roi_slicer is not None:
        slicer = roi_slicer
    elif mask is not None:
        slicer = mask
    else:
        slicer = slice(None)
    current_mean = img[slicer].mean()
    current_std = img[slicer].std()
    target_mean = reference[slicer].mean()
    target_std = reference[slicer].std()
    return (img - current_mean) * target_std / current_std + target_mean


def image_to_object(
    img_mag: Union[torch.Tensor, float] = None, 
    img_phase: Union[torch.Tensor, float] = None, 
) -> torch.Tensor:
    """Convert a pixel-space image to a complex object tensor.
    
    Parameters
    ----------
    img_mag: torch.Tensor
        A (n_slices, 3, h, w) tensor giving the magnitude of the image. The image is expected
        to be scaled to the usual range of object magnitudes, e.g., around 1.
    img_phase: torch.Tensor
        A (n_slices, 3, h, w) tensor giving the phase of the image. The image is expected
        to be scaled to the usual range of object phases, e.g., centered around 0.
        
    Returns
    -------
    torch.Tensor
        A (n_slices, h, w) tensor giving the complex object.
    """
    if img_mag is None and img_phase is None:
        raise ValueError("Either img_mag or img_phase must be provided.")
    
    if img_mag is None:
        mag = 1
    elif isinstance(img_mag, (torch.Tensor, np.ndarray)):
        mag = img_mag.mean(1)
    else:
        mag = img_mag
        
    if img_phase is None:
        phase = 0
    elif isinstance(img_phase, (torch.Tensor, np.ndarray)):
        phase = img_phase.mean(1)
    else:
        phase = img_phase
        
    obj = mag * torch.exp(1j * phase)
    obj = maths.TypeCastFunction.apply(obj, torch.complex64)
    return obj


def object_to_image(
    obj: torch.Tensor,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """Convert a complex object tensor to an image assumed by the encoder.
    
    Parameters
    ----------
    obj: torch.Tensor
        A (n_slices, h, w) tensor giving the complex object.
        
    Returns
    -------
    torch.Tensor
        A (n_slices, 3, h, w) tensor giving the image.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    mag = obj.abs()
    img_mag = mag.unsqueeze(1)
    img_mag = img_mag.repeat(1, 3, 1, 1)
    img_mag = img_mag.to(dtype)
    phase = obj.angle()
    img_phase = phase.unsqueeze(1)
    img_phase = img_phase.repeat(1, 3, 1, 1)
    img_phase = img_phase.to(dtype)
    return img_mag, img_phase


def pil_image_to_tensor(
    pil_image: Image.Image, 
    dtype: torch.dtype = None, 
    device: torch.device = None
) -> torch.Tensor:
    """Convert a 8-bit PIL image to a tensor in range [0, 1] with shape (1, 3, h, w).
    
    Parameters
    ----------
    pil_image: PIL.Image.Image
        The image to convert.
    dtype: torch.dtype
        The dtype of the output tensor.
    device: torch.device
        The device of the output tensor.
        
    Returns
    -------
    torch.Tensor
        A (1, 3, h, w) tensor giving the image.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    img = torch.tensor(np.array(pil_image), dtype=dtype, device=device)
    img = img / 255.0
    img = img.permute(2, 0, 1)[None, ...]
    return img
