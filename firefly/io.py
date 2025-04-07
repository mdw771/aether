import os
import logging
from typing import Optional

import torch
import diffusers

import firefly
import firefly.util as util


logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, img2img: bool = False, device: str = "cuda", *args, **kwargs):
        self.device = device
        self.img2img = img2img
        
    @staticmethod
    def get_local_model_repository_path():
        return os.path.join(firefly.__path__[0], os.pardir, "models")
    

class HuggingFaceModelLoader(ModelLoader):
    def __init__(
        self, 
        model_path: str, 
        pipe_class: type[diffusers.DiffusionPipeline] = diffusers.DiffusionPipeline,
        pipe_kwargs: Optional[dict] = None, 
        *args, **kwargs
    ):
        """HuggingFace model loader.
        
        Parameters
        ----------
        model_path: str
            The path to model. The path should be given as a Hugging Face model 
            name or path so that it can be fetched from Hugging Face --
            for example, "stabilityai/stable-diffusion-3.5-medium". If the model
            exists in `<firefly_root>/models/`, it will be loaded from there.
            Otherwise, it will be downloaded from Hugging Face and saved to that
            location.
        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.pipe_class = pipe_class
        self.pipe_kwargs = pipe_kwargs if pipe_kwargs is not None else {}
        self.local_model_path = os.path.join(self.get_local_model_repository_path(), self.model_path)
        
    def load(self) -> None:
        if self.model_path is not None and os.path.exists(self.local_model_path):
            logger.info(f"Loading model from {self.local_model_path}")
            self.load_local_model()
        else:
            logger.info(f"Downloading model from Hugging Face to {self.model_path}")
            self.download_model()
            self.save_model()
        self.pipe = self.pipe.to(self.device)
                
    def load_local_model(self):
        with util.ignore_default_device():
            self.pipe = self.pipe_class.from_pretrained(
                self.local_model_path,
                **self.pipe_kwargs
            )

    def download_model(self):
        with util.ignore_default_device():
            self.pipe = self.pipe_class.from_pretrained(
                self.model_path,
                **self.pipe_kwargs
            )
    
    def save_model(self):
        self.pipe.save_pretrained(self.local_model_path)
    
    
class HuggingFaceMultiModelLoader(ModelLoader):
    def __init__(
        self, 
        model_paths: list[str], 
        model_loader_classes: list[type[ModelLoader]],
        pipe_kwargs_list: Optional[list[dict]] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_paths = model_paths
        self.model_loader_classes = model_loader_classes
        self.pipe_kwargs_list = pipe_kwargs_list if pipe_kwargs_list is not None else [None] * len(model_paths)
        self.loaders = []
        self.pipes = None
        
    def load(self) -> None:
        for i, model_path in enumerate(self.model_paths):
            self.loaders.append(self.model_loader_classes[i](model_path, self.pipe_kwargs_list[i]))
            self.pipes.append(self.loaders[-1].pipe)

        
    def download_model(self):
        with util.ignore_default_device():
            if self.img2img:
                self.pipe = diffusers.AutoPipelineForImage2Image.from_pretrained(
                    self.model_path,
                    **self.pipe_kwargs
                )
            else:
                self.pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
                    self.model_path,
                    **self.pipe_kwargs
                )

    def save_model(self):
        for i, pipe in enumerate(self.pipes):
            pipe.save_pretrained(
                os.path.join(self.get_local_model_repository_path(), self.model_paths[i])
            )
        
        
class HuggingFaceDeepFloydIFModelLoader(HuggingFaceMultiModelLoader):
    def __init__(
        self, 
        *args, **kwargs
    ):
        super().__init__(
            model_paths=(
                "DeepFloyd/IF-I-XL-v1.0", 
                "DeepFloyd/IF-II-L-v1.0", 
                "stabilityai/stable-diffusion-x4-upscaler"
            ), 
            model_loader_classes=(
                HuggingFaceModelLoader, 
                HuggingFaceModelLoader, 
                HuggingFaceModelLoader
            ),
            pipe_kwargs_list=(
                {"torch_dtype": torch.float16},
                {"torch_dtype": torch.float16, "text_encoder": None},
                {"torch_dtype": torch.float16, "feature_extractor": None, "safety_checker": None, "watermarker": None}
            ),
            *args, **kwargs
        )
        
    def load(self) -> None:
        # stage 1
        stage_1 = diffusers.DiffusionPipeline.from_pretrained(
            os.path.join(self.get_local_model_repository_path(), "DeepFloyd/IF-I-XL-v1.0"), 
            torch_dtype=torch.float16
        )
        # stage_1.enable_model_cpu_offload()

        # stage 2
        stage_2 = diffusers.DiffusionPipeline.from_pretrained(
            os.path.join(self.get_local_model_repository_path(), "DeepFloyd/IF-II-L-v1.0"), 
            text_encoder=None, torch_dtype=torch.float16
        )
        # stage_2.enable_model_cpu_offload()
        # stage 3
        safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = diffusers.DiffusionPipeline.from_pretrained(
            os.path.join(self.get_local_model_repository_path(), "stabilityai/stable-diffusion-x4-upscaler"), 
            **safety_modules, 
            torch_dtype=torch.float16
        )
        # stage_3.enable_model_cpu_offload()
        
        self.pipes = [stage_1, stage_2, stage_3]
    