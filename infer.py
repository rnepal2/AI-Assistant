''''
    Inference from Huggingface Stable Diffusion Model
'''
import os
import torch
from IPython.utils import io
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import schedulers
from PIL import Image
 
class cfg:
    token = os.getenv("HF_USER_TOKEN") 
    checkpoint = "runwayml/stable-diffusion-v1-5"
    seed = 12345
 
scheduler_params_default = {
                            "beta_start": 0.0001,
                            "beta_end": 0.02,
                            "beta_schedule": "scaled_linear",
                            "num_train_timesteps": 1000
                           }    

schedulers_dict = {
                    "DDIM": schedulers.scheduling_ddim.DDIMScheduler,
                    "PNDM": schedulers.scheduling_pndm.PNDMScheduler,
                    "LMSD": schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
                }

class Inference:
    
    assert torch.cuda.is_available() == True
    generator = torch.Generator(device="cuda")
    
    def __init__(self, scheduler="LMSD", scheduler_params=None):
        
        if not scheduler_params:
            self.scheduler_params = scheduler_params_default
        else:
            sparams = scheduler_params_default.copy()
            for k, v in scheduler_params.items():
                sparams[k] = v
            self.scheduler_params = sparams
            
        self.scheduler = scheduler
        scheduler_obj = schedulers_dict[self.scheduler](**self.scheduler_params)
            
        self.pipeline = StableDiffusionPipeline.from_pretrained(cfg.checkpoint, torch_dtype=torch.float16, revision="fp16", use_auth_token=cfg.token)
        self.pipeline.to("cuda")
        self.pipeline.enable_attention_slicing()
        
    def model_input(self, seed):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        _inputs = {
                    "eta": 0.0,
                    "generator": generator.manual_seed(seed),
                    "guidance_scale": 7.5,
                    "height": 512,
                    "width": 512,
                    "num_inference_steps": 100,
                    "output_type": "pil",   
                }
        return _inputs
    
    def __call__(self, prompt, seed=cfg.seed, params=None):
        if not params:
            _params = self.model_input(seed)
        else:
            _params = self.model_input(seed)
            for p, v in params.items():
                _params[p] = v
        with io.capture_output() as captured:
            image = self.pipeline(prompt, **_params).images[0]
        return image 

