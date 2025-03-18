import logging

import torch
from ptychi.api.task import PtychographyTask
from ptychi.api.options.task import PtychographyTaskOptions
from ptychi.data_structures.parameter_group import PtychographyParameterGroup
from ptychi.data_structures.object import PlanarObject
from ptychi.utils import to_tensor

from firefly.reconstructor import GuidedDiffusionReconstructor, GuidedFlowMatchingReconstructor
import firefly.io as fio


logger = logging.getLogger(__name__)


class GuidedDiffusionPtychographyTask(PtychographyTask):
    
    def __init__(self, options: PtychographyTaskOptions):
        self.model_loader = None
        super().__init__(options=options)
        
    def build(self):
        self.build_model_loader()
        super().build()
        
    def build_object(self):
        data = to_tensor(self.object_options.initial_guess)
        kwargs = {
            "data": data,
            "options": self.object_options,
            "data_as_parameter": False
        }
        self.object = PlanarObject(**kwargs)
        self.object.optimizable = False
        
    def build_model_loader(self):
        self.model_loader = fio.HuggingFaceStableDiffusionModelLoader(
            model_path=self.reconstructor_options.model_path,
            device=torch.get_default_device(),
        )
        
    def build_reconstructor(self):
        par_group = PtychographyParameterGroup(
            object=self.object,
            probe=self.probe,
            probe_positions=self.probe_positions,
            opr_mode_weights=self.opr_mode_weights,
        )

        reconstructor_class = self.get_reconstructor_class()
        logger.info(f"Using {reconstructor_class.__name__}.")
        
        self.reconstructor = reconstructor_class(
            parameter_group=par_group,
            dataset=self.dataset,
            model_loader=self.model_loader,
            options=self.reconstructor_options,
        )
        self.reconstructor.build()
        
    def get_reconstructor_class(self):
        if "stable-diffusion-3" in self.reconstructor_options.model_path:
            return GuidedFlowMatchingReconstructor
        else:
            return GuidedDiffusionReconstructor
