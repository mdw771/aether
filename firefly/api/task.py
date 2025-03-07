import torch
from ptychi.api.task import PtychographyTask
from ptychi.api.options.task import PtychographyTaskOptions
from ptychi.data_structures.parameter_group import PtychographyParameterGroup

from firefly.reconstructor import GuidedDiffusionReconstructor
import firefly.io as fio


class GuidedDiffusionPtychographyTask(PtychographyTask):
    
    def __init__(self, options: PtychographyTaskOptions):
        self.model_loader = None
        super().__init__(options=options)
        
    def build(self):
        self.build_model_loader()
        super().build()
        
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

        self.reconstructor = GuidedDiffusionReconstructor(
            parameter_group=par_group,
            dataset=self.dataset,
            model_loader=self.model_loader,
            options=self.reconstructor_options,
        )
        self.reconstructor.build()
