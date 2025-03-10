import os
import logging

import torch
import diffusers

import firefly
import firefly.util as util


logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, device: str = "cuda", *args, **kwargs):
        self.device = device
    
    @staticmethod
    def get_local_model_repository_path():
        return os.path.join(firefly.__path__[0], os.pardir, "models")
    

class HuggingFaceModelLoader(ModelLoader):
    def __init__(self, model_path: str, *args, **kwargs):
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
        self.local_model_path = os.path.join(self.get_local_model_repository_path(), self.model_path)
        
    def load(self) -> None:
        if self.model_path is not None and os.path.exists(self.local_model_path):
            logger.info(f"Loading model from {self.local_model_path}")
            self.load_local_model()
        else:
            logger.info(f"Downloading model from Hugging Face to {self.model_path}")
            self.download_model()
            self.save_model()
                
    def load_local_model(self):
        raise NotImplementedError("Local model loading is not implemented")
    
    def download_model(self):
        raise NotImplementedError("Model downloading is not implemented")
    
    def save_model(self):
        raise NotImplementedError("Model saving is not implemented")


class HuggingFaceStableDiffusionModelLoader(HuggingFaceModelLoader):
    def __init__(
        self, 
        model_path: str = "stabilityai/stable-diffusion-3.5-medium",
        *args, **kwargs
    ):
        super().__init__(model_path, *args, **kwargs)
        self.pipe = None
        
    def load(self) -> None:
        super().load()
        
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()
        
    def get_pipe_kwargs(self):
        return {
            "torch_dtype": torch.float16,
            "use_safetensors": True
        }
        
    def load_local_model(self):
        with util.ignore_default_device():
            self.pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
                self.local_model_path,
                **self.get_pipe_kwargs()
            )
        
    def download_model(self):
        with util.ignore_default_device():
            self.pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
                self.model_path,
                **self.get_pipe_kwargs()
            )

    def save_model(self):
        self.pipe.save_pretrained(self.local_model_path)


if __name__ == "__main__":
    model_loader = HuggingFaceStableDiffusionModelLoader(
        model_path="stabilityai/stable-diffusion-xl-base-1.0"
    )
    model_loader.load()
    pipe = model_loader.pipe

    with torch.inference_mode():
        image = pipe(
            "A capybara holding a sign that reads Hello World",
            num_inference_steps=40,
            guidance_scale=4.5,
        ).images[0]

        # Save the generated image
        image.show()
