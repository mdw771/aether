# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/aether/blob/main/LICENSE

import numpy as np
import dtcwt

from ptychi.utils import to_numpy, to_tensor

from aether.reconstructors.pnp import PnPReconstructor
from aether.api.options.pnp_xblur import CWTBlurryFeatureRemovalOptions


class PnPXBlurDTCWTReconstructor(PnPReconstructor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.options.prior_projection_options, CWTBlurryFeatureRemovalOptions
        )
    
    @staticmethod
    def dtcwt_transform_multislice(
        obj: np.ndarray,
        n_levels: int = 2,
    ) -> list:
        """Apply dual-tree complex wavelet transform to a multislice object.
        
        This method returns a list of transform result objects, each for a
        slice. The structure of each transform result is as follows:
        
        - lowpass
          - [h_{n-1}, w_{n-1}]
        - highpasses
          - level_1
            - [h_1, w_1, 6] of 6 filters
          - level_2
            - [h_2, w_2, 6] of 6 filters
          - ...
          - level_n
            - [h_n, w_n, 6] of 6 filters
        
        Parameters
        ----------
        obj : np.ndarray
            A (n_slices, h, w) array of the multislice object to transform.
        n_levels : int, optional
            The number of levels of the transform.
            
        Returns
        -------
        transform_results : list
            A list of dtcwt transform results, each corresponding to a slice
            of the multislice object.
        """
        transform = dtcwt.Transform2d()
        transform_results = []
        for i_slice in range(obj.shape[0]):
            slice_transform_result = transform.forward(obj[i_slice], nlevels=n_levels)
            transform_results.append(slice_transform_result)
        return transform_results
    
    @staticmethod
    def process_dtcwt_coefficients(
        transform_results: list,
        attenuation_factor: float = 0.1,
    ) -> list:
        """Process the coefficients of the transform results.
        
        Parameters
        ----------
        transform_results : list
            A list of transform results, each corresponding to a slice.
        attenuation_factor : float, optional
            The factor to attenuate the coefficients.
            
        Returns
        -------
        transform_results : list
            A list of transform results, each corresponding to a slice.
        """
        n_slices = len(transform_results)
        n_levels = len(transform_results[0].highpasses)
        
        # Process highpasses
        for i_level in range(n_levels):
            for i_filter in range(transform_results[0].highpasses[i_level].shape[-1]):
                coeff_stack = [
                    transform_results[i_slice].highpasses[i_level][:, :, i_filter]
                    for i_slice in range(n_slices)
                ]
                
                # Attenuate anything that is less than the max absolute coefficient.
                coeff_stack = np.stack(coeff_stack, axis=0)
                max_map = np.max(np.abs(coeff_stack), axis=0)
                coeff_stack[np.abs(coeff_stack) < max_map] *= attenuation_factor
                
                # Put them back
                for i_slice in range(n_slices):
                    transform_results[i_slice].highpasses[i_level][:, :, i_filter] = coeff_stack[i_slice]
        return transform_results
    
    @staticmethod
    def dtcwt_inverse_transform_multislice(
        transform_results: list,
    ) -> np.ndarray:
        """Inverse transform the transform results.
        """
        transform = dtcwt.Transform2d()
        inv_transform_results = []
        for i_slice in range(len(transform_results)):
            slice_inv_transform_result = transform.inverse(transform_results[i_slice])
            inv_transform_results.append(slice_inv_transform_result)
        return np.stack(inv_transform_results, axis=0)
    
    def project_to_prior(self):
        obj = self.x + self.u
        obj = to_numpy(obj)
        
        transform_results = self.dtcwt_transform_multislice(
            obj, n_levels=self.options.prior_projection_options.n_levels
        )
        transform_results = self.process_dtcwt_coefficients(transform_results)
        obj = self.dtcwt_inverse_transform_multislice(transform_results)
        
        self.v = to_tensor(obj)
        self.num_prior_projections += 1
        