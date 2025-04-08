import torch


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


def match_mean_std(img: torch.Tensor, reference: torch.Tensor, roi_slicer: tuple[slice, ...]):
    """Match the mean and std of the image to the target mean and std within the ROI.
    """
    current_mean = img[roi_slicer].mean()
    current_std = img[roi_slicer].std()
    target_mean = reference[roi_slicer].mean()
    target_std = reference[roi_slicer].std()
    return (img - current_mean) * target_std / current_std + target_mean
