import logging

import torch
import diffusers

from ptychi.api.task import PtychographyTask
from ptychi.api.options.task import PtychographyTaskOptions
from ptychi.data_structures.parameter_group import PtychographyParameterGroup
from ptychi.data_structures.object import PlanarObject
from ptychi.utils import to_tensor
from ptychi.reconstructors.base import Reconstructor
import ptychi.maps as maps
import ptychi.utils as utils

import aether.reconstructors.pnp
import aether.io as fio
import aether.api as api

logger = logging.getLogger(__name__)


class PnPPtychographyTask(PtychographyTask):
    
    def __init__(self, options: PtychographyTaskOptions):
        self.model_loader = None
        
        self.options = options
        self.object_options = options.object_options
        self.data_options = options.data_options
        self.reconstructor_options = options.reconstructor_options

        self.object = None
        self.reconstructor: Reconstructor | None = None

        self.check_options()
        self.build()
        
    def build(self):
        self.build_model_loader()
        self.build_default_device()
        self.build_default_dtype()
        self.build_object()
        self.build_reconstructor()
        
    def build_object(self):
        data = to_tensor(self.object_options.initial_guess)
        kwargs = {
            "data": data,
            "options": self.object_options,
            "data_as_parameter": False
        }
        self.object = PlanarObject(**kwargs)
        self.object.optimizable = False
        
    def build_default_dtype(self):
        torch.set_default_dtype(maps.get_dtype_by_enum(self.reconstructor_options.default_dtype))
        utils.set_default_complex_dtype(
            maps.get_complex_dtype_by_enum(self.reconstructor_options.default_dtype)
        )
        
    def build_model_loader(self):
        if isinstance(self, PnPPtychographyTask):
            img2img = True
        else:
            img2img = False
        if (
            hasattr(self.reconstructor_options, "prior_projection_options") 
            and (hasattr(self.reconstructor_options.prior_projection_options, "model_path"))
        ):
            model_path = self.reconstructor_options.prior_projection_options.model_path
        elif hasattr(self.reconstructor_options, "model_path"):
            model_path = self.reconstructor_options.model_path
        else:
            return
        if "deepfloyd" in model_path.lower():
            logger.warning(
                "DeepFloyd will be loaded from default locations and with default variants. "
                "The exact path provided will be ignored."
            )
            self.model_loader = fio.HuggingFaceDeepFloydIFModelLoader(
                device=torch.get_default_device(),
                img2img=img2img,
            )
        else:
            if isinstance(self.options.reconstructor_options.prior_projection_options, api.LEDITSPPOptions):
                pipe_class = diffusers.LEditsPPPipelineStableDiffusion
            else:
                pipe_class = diffusers.DiffusionPipeline
            self.model_loader = fio.HuggingFaceModelLoader(
                model_path=model_path,
                device=torch.get_default_device(),
                img2img=img2img,
                pipe_class=pipe_class,
                pipe_kwargs={"torch_dtype": torch.float16}
            )

    def build_reconstructor(self):
        PtychographyParameterGroup.__post_init__ = lambda self: None
        par_group = PtychographyParameterGroup(
            object=self.object,
            probe=None,
            probe_positions=None,
            opr_mode_weights=None
        )

        reconstructor_class = self.select_reconstructor_class()
        logger.info(f"Using {reconstructor_class.__name__}.")
        
        self.reconstructor = reconstructor_class(
            parameter_group=par_group,
            model_loader=self.model_loader,
            options=self.reconstructor_options,
        )
        self.reconstructor.build()
        
    def select_reconstructor_class(self) -> type[aether.reconstructors.pnp.PnPReconstructor]:
        if isinstance(self.reconstructor_options.prior_projection_options, api.LEDITSPPOptions):
            return aether.reconstructors.pnp.PnPLEDITSPPReconstructor
        else:
            raise ValueError(
                "Unable to infer reconstructor class from reconstructor_options.prior_projection_options."
            )
