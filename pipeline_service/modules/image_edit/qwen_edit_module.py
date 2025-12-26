import math
from os import PathLike
from pathlib import Path
from typing import Optional, Any, Literal
from safetensors import safe_open
import torch
from pydantic import BaseModel, Field
from diffusers import QwenImageEditPlusPipeline
import time
from PIL import Image

import json

from dotenv import load_dotenv

from schemas.custom_types import BFloatTensor, IntTensor
load_dotenv()

from logger_config import logger
import hashlib

from diffusers.models import QwenImageTransformer2DModel
from modules.image_edit.qwen_manager import QwenManager
from config import Settings

class EmbeddedPrompting(BaseModel):
    prompt_embeds: BFloatTensor
    prompt_embeds_mask: Optional[IntTensor] = None

class TextPrompting(BaseModel):
    prompt: str = Field(alias="positive")
    negative_prompt: Optional[str] = Field(default=None, alias="negative")

class QwenEditModule(QwenManager):
    """Qwen module for image editing operations."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._empty_image = Image.new('RGB', (1024, 1024))

        self.base_model_path = settings.qwen_edit_base_model_path
        self.edit_model_path = settings.qwen_edit_model_path
        self.prompt_path = settings.qwen_edit_prompt_path
        self.mv_prompt_path = settings.qwen_mv_prompt_path
        self.view_prompts_path = settings.qwen_view_prompts_path
        self.prompting = self._set_prompting()
        self.mv_prompting = None  # Lazy load when needed
        self.view_prompts = None  # Lazy load when needed

        self.pipe_config = {
            "num_inference_steps": settings.num_inference_steps,
            "true_cfg_scale": settings.true_cfg_scale,
            "height": settings.qwen_edit_height,
            "width": settings.qwen_edit_width,

        }

    def _set_text_prompting(self, path: Optional[PathLike] = None) -> TextPrompting:
        path = path or self.prompt_path
        with open(path, "r") as f:
            edit_prompt = TextPrompting.model_validate_json(json.dumps(json.load(f)))
            return edit_prompt


    def _set_embedded_prompting(self, path: Optional[PathLike] = None) -> EmbeddedPrompting:
        path = path or self.prompt_path
        with safe_open(path,framework="pt", device=self.device)as f:
            tensors = {key: f.get_tensor(key) for key in f.keys()}
            embedding = EmbeddedPrompting(**tensors)
        return embedding
    
    def _set_prompting(self, path: Optional[PathLike] = None) -> TextPrompting | EmbeddedPrompting:
        path = Path(path or self.prompt_path)
        if path.suffix == ".safetensors":
            return self._set_embedded_prompting(path)
        else:
            return self._set_text_prompting(path)

    def _get_model_transformer(self):
        """Load the Nunchaku Qwen transformer for image editing."""
        return  QwenImageTransformer2DModel.from_pretrained(
                self.edit_model_path,
                subfolder="transformer",
                torch_dtype=self.dtype
            )

    def _get_model_pipe(self, transformer, scheduler):

        return QwenImageEditPlusPipeline.from_pretrained(
                self.edit_model_path,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=self.dtype
            )

    def _get_scheduler_config(self):
        """Return scheduler configuration for image editing."""
        return  {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }

    def _prepare_input_image(self, image: Image, megapixels: float = 1.0):
        total = int(megapixels * 1024 * 1024)

        scale_by = math.sqrt(total / (image.width * image.height))
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)

        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _run_model_pipe(self, seed: Optional[int] = None, **kwargs):
        if seed:
            kwargs.update(dict(generator=torch.Generator(device=self.device).manual_seed(seed)))
        image = kwargs.pop("image", self._empty_image)
        result = self.pipe(
                image=image,
                **self.pipe_config,
                **kwargs)
        return result
    
    def _run_edit_pipe(self,
                       prompt_image: Image.Image,
                       seed: Optional[int] = None,
                       **kwargs):
        prompt_image = self._prepare_input_image(prompt_image)
        return self._run_model_pipe(seed=seed, image=prompt_image, **kwargs)
    
    
    def edit_image(self, prompt_image: Image.Image, seed: int):
        """ 
        Edit the image using Qwen Edit.

        Args:
            prompt_image: The prompt image to edit.
            reference_image: The reference image to edit.

        Returns:
            The edited image.
        """
        if self.pipe is None:
            logger.error("Edit Model is not loaded")
            raise RuntimeError("Edit Model is not loaded")
        
        try:
            start_time = time.time()

            prompting = self.prompting.model_dump()
            
            # Run the edit pipe
            result = self._run_edit_pipe(prompt_image=prompt_image,
                                         **prompting,
                                         seed=seed)
            
            generation_time = time.time() - start_time
            
            image_edited = result.images[0]
            
            logger.success(f"Edited image generated in {generation_time:.2f}s, Size: {image_edited.size}, Seed: {seed}")
            
            return image_edited
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e

    def edit_image_mv(self, prompt_image: Image.Image, seed: int, grid_size_multiplier: int = 1):
        """ 
        Edit the image using Qwen Edit to generate a multi-view grid.

        Args:
            prompt_image: The prompt image to edit into a grid.
            seed: Random seed for generation.
            grid_size_multiplier: Multiplier for grid size (1 = 2x2 grid at 1x resolution).
                                 Default 1 means 1024x1024 grid, cropped to 512x512 views (scaled up later).

        Returns:
            The edited grid image with 4 views.
        """
        if self.pipe is None:
            logger.error("Edit Model is not loaded")
            raise RuntimeError("Edit Model is not loaded")
        
        try:
            start_time = time.time()

            # Lazy load MV prompting if not already loaded
            if self.mv_prompting is None:
                self.mv_prompting = self._set_prompting(self.mv_prompt_path)
            
            prompting = self.mv_prompting.model_dump()
            
            # Log the prompt being used for debugging
            prompt_text = prompting.get('prompt', prompting.get('positive', 'N/A'))
            logger.info(f"Using MV prompt (first 200 chars): {prompt_text[:200] if isinstance(prompt_text, str) else 'N/A'}...")
            logger.debug(f"MV prompt keys: {list(prompting.keys())}")
            
            # Generate grid at 1x resolution (1024x1024), cropped views will be 512x512 and scaled up later
            grid_width = self.pipe_config["width"] * grid_size_multiplier
            grid_height = self.pipe_config["height"] * grid_size_multiplier
            
            # Temporarily override pipe config for grid generation
            original_width = self.pipe_config["width"]
            original_height = self.pipe_config["height"]
            self.pipe_config["width"] = grid_width
            self.pipe_config["height"] = grid_height
            
            try:
                # Run the edit pipe with MV prompt at 2x resolution
                result = self._run_edit_pipe(prompt_image=prompt_image,
                                             **prompting,
                                             seed=seed)
            except Exception as e:
                # Ensure config is restored even if there's an error
                self.pipe_config["width"] = original_width
                self.pipe_config["height"] = original_height
                raise e
            finally:
                # Restore original config (safety net)
                self.pipe_config["width"] = original_width
                self.pipe_config["height"] = original_height
            
            generation_time = time.time() - start_time
            
            grid_image = result.images[0]
            
            logger.success(f"Multi-view grid image generated in {generation_time:.2f}s, Size: {grid_image.size}, Seed: {seed}")
            
            return grid_image
            
        except Exception as e:
            logger.error(f"Error generating multi-view grid image: {e}")
            raise e

    def _load_view_prompts(self) -> dict:
        """Load view-specific prompts from JSON file."""
        if self.view_prompts is None:
            try:
                with open(self.view_prompts_path, 'r', encoding='utf-8') as f:
                    self.view_prompts = json.load(f)
                logger.info(f"Loaded view prompts from {self.view_prompts_path}")
            except Exception as e:
                logger.error(f"Failed to load view prompts from {self.view_prompts_path}: {e}")
                # Fallback to default prompts
                self.view_prompts = {
                    "front_view": {"positive": "Show this exact same object from the front view, facing the camera directly. Keep the exact same object type, design, materials, colors, and style. Clean white background."},
                    "back_view": {"positive": "Show this exact same object from the back view, opposite side. Keep the exact same object type, design, materials, colors, and style. Clean white background."},
                    "side_view": {"positive": "Show this exact same object from the side view, 90-degree profile angle. Keep the exact same object type, design, materials, colors, and style. Clean white background."},
                    "isometric_view": {"positive": "Show this exact same object from an isometric three-quarter view, 45-degree angle. Keep the exact same object type, design, materials, colors, and style. Clean white background."}
                }
        return self.view_prompts

    def generate_sequential_views(self, prompt_image: Image.Image, seed: int, view_count: int = 3) -> dict[str, Image.Image]:
        """
        Generate sequential views of the object instead of using grid cropping.
        
        Args:
            prompt_image: Input image to generate views from
            seed: Random seed for generation
            view_count: Number of additional views to generate (2, 3, or 4)
            
        Returns:
            Dictionary mapping view names to generated images
        """
        try:
            start_time = time.time()
            logger.info(f"Generating {view_count} sequential views with seed {seed}")
            
            # Load view prompts
            view_prompts = self._load_view_prompts()
            
            # Define view sequences based on count
            view_sequences = {
                2: ["front_view", "back_view"],
                3: ["front_view", "back_view", "side_view"], 
                4: ["front_view", "back_view", "side_view", "isometric_view"]
            }
            
            if view_count not in view_sequences:
                raise ValueError(f"Unsupported view_count: {view_count}. Must be 2, 3, or 4.")
            
            views_to_generate = view_sequences[view_count]
            generated_views = {}
            
            # Generate each view sequentially
            for i, view_name in enumerate(views_to_generate):
                view_seed = seed + i + 1  # Use different seed for each view
                view_prompt = view_prompts.get(view_name, {})
                
                if not view_prompt:
                    logger.warning(f"No prompt found for view: {view_name}, skipping")
                    continue
                
                logger.info(f"Generating {view_name} (seed: {view_seed})")
                
                try:
                    # Convert view prompt to proper format using TextPrompting
                    text_prompting = TextPrompting.model_validate(view_prompt)
                    prompting_dict = text_prompting.model_dump()
                    
                    # Generate the view using the specific prompt
                    result = self._run_edit_pipe(
                        prompt_image=prompt_image,
                        **prompting_dict,
                        seed=view_seed
                    )
                    
                    generated_views[view_name] = result.images[0]
                    logger.info(f"Generated {view_name} successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {view_name}: {e}")
                    # Continue with other views even if one fails
                    continue
            
            generation_time = time.time() - start_time
            logger.success(f"Generated {len(generated_views)} sequential views in {generation_time:.2f}s")
            
            return generated_views
            
        except Exception as e:
            logger.error(f"Error generating sequential views: {e}")
            raise e