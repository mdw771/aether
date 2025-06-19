import logging

import torch
from ptychi.api.task import PtychographyTask
from ptychi.api.options.task import PtychographyTaskOptions
from ptychi.data_structures.parameter_group import PtychographyParameterGroup
from ptychi.data_structures.object import PlanarObject
from ptychi.utils import to_tensor

import diffusers
import aether.api as api
from aether.reconstructors.guided_sampling import (
    GuidedLatentDiffusionReconstructor, 
    GuidedLatentFlowMatchingReconstructor, 
    GuidedDeepFloydIFReconstructor,
)
from aether.reconstructors.latent_admmdiff import ADMMDiffReconstructor
from aether.reconstructors.latent_dps import LatentDPSReconstructor
from aether.reconstructors.alt_proj import (
    AlternatingProjectionReconstructor
)
import aether.io as fio


logger = logging.getLogger(__name__)


class BaseDiffusionPtychographyTask(PtychographyTask):
    
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
        if isinstance(self, AlternatingProjectionDiffusionPtychographyTask):
            img2img = True
        else:
            img2img = False
        if "deepfloyd" in self.reconstructor_options.model_path.lower():
            logger.warning(
                "DeepFloyd will be loaded from default locations and with default variants. "
                "The exact path provided will be ignored."
            )
            self.model_loader = fio.HuggingFaceDeepFloydIFModelLoader(
                device=torch.get_default_device(),
                img2img=img2img,
            )
        else:
            if isinstance(self, AlternatingProjectionDiffusionPtychographyTask):
                pipe_class = diffusers.LEditsPPPipelineStableDiffusion
            else:
                pipe_class = diffusers.DiffusionPipeline
            self.model_loader = fio.HuggingFaceModelLoader(
                model_path=self.reconstructor_options.model_path,
                device=torch.get_default_device(),
                img2img=img2img,
                pipe_class=pipe_class,
                pipe_kwargs={"torch_dtype": torch.float16}
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


class GuidedDiffusionPtychographyTask(BaseDiffusionPtychographyTask):
        
    def get_reconstructor_class(self):
        if isinstance(self.options.reconstructor_options, api.ADMMDiffReconstructorOptions):
            return ADMMDiffReconstructor
        if isinstance(self.options.reconstructor_options, api.LatentDPSReconstructorOptions):
            return LatentDPSReconstructor
        if "stable-diffusion-3" in self.reconstructor_options.model_path.lower():
            return GuidedLatentFlowMatchingReconstructor
        elif "deepfloyd" in self.reconstructor_options.model_path.lower():
            return GuidedDeepFloydIFReconstructor
        else:
            return GuidedLatentDiffusionReconstructor


class AlternatingProjectionDiffusionPtychographyTask(BaseDiffusionPtychographyTask):
        
    def get_reconstructor_class(self):
        return AlternatingProjectionReconstructor
