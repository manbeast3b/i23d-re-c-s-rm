from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional, List

from PIL import Image
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse, TrellisParams, TrellisRequest, TrellisResult
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.gs_generator.reconviagen_manager import ReconViaGenManager
from modules.utils import secure_randint, set_random_seed, decode_image, to_png_base64, save_files, crop_grid_image, save_image


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)
        self.reconviagen = ReconViaGenManager(settings)

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        
        # Only start one 3D generation service based on settings
        if self.settings.use_reconviagen:
            logger.info("Using ReconViaGen for 3D generation")
            await self.reconviagen.startup()
        else:
            logger.info("Using standard Trellis for 3D generation")
            await self.trellis.startup()
        
        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        
        # Shutdown the active 3D generation service
        if self.settings.use_reconviagen:
            await self.reconviagen.shutdown()
        else:
            await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(64,64),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_imge_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_imge_bytes,seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            
        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create request
        request = GenerateRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )
        
        # Generate
        response = await self.generate_gs(request)
        
        # Return binary PLY
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")
            
        return response.ply_file_base64 # bytes

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.
        
        Args:
            request: Generation request with prompt and settings
            
        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        image_edited = self.qwen_edit.edit_image(prompt_image=image, seed=request.seed)

        # Create timestamp for this generation (used for saving intermediate images)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save image_edited
        # save_image(image_edited, "png", "image_edited", timestamp)
        # logger.info(f"Saved image_edited to generated_outputs/png/{timestamp}/image_edited.png")

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params if request.trellis_params else TrellisParams.from_settings(self.settings)
        
        # Force single-view mode for goat-re-conseq-single
        # Always use single image, no multi-view generation
        use_multi_view = False

        trellis_result: Optional[TrellisResult] = None

        if use_multi_view:
            # Multi-view flow: Sequential view generation → Remove backgrounds → Multi-view ReconViaGen/Trellis
            logger.info("Using multi-view generation pipeline")
            
            # Check if sequential view generation is enabled
            use_sequential = getattr(self.settings, 'use_sequential_views', True)
            view_count = getattr(self.settings, 'multiview_count', 3)
            
            if use_sequential:
                # 2a. Generate sequential views using individual prompts
                logger.info(f"Generating {view_count} sequential views")
                sequential_views = self.qwen_edit.generate_sequential_views(
                    prompt_image=image_edited, 
                    seed=request.seed, 
                    view_count=view_count
                )
                
                # Add the original edited image as the first view
                all_views = {"original": image_edited}
                all_views.update(sequential_views)
                
                logger.info(f"Generated {len(sequential_views)} sequential views: {list(sequential_views.keys())}")
                
                # Save each generated view
                # for view_name, view_img in all_views.items():
                #     save_image(view_img, "png", f"sequential_{view_name}", timestamp)
                #     logger.info(f"Saved sequential_{view_name} to generated_outputs/png/{timestamp}/sequential_{view_name}.png")
                
                # 2b. Remove background from each sequential view
                images_without_background = []
                for view_name, view_img in all_views.items():
                    bg_removed = self.rmbg.remove_background(view_img)
                    images_without_background.append(bg_removed)
                    # Save background-removed image
                    # save_image(bg_removed, "png", f"bg_removed_{view_name}", timestamp)
                    # logger.info(f"Saved bg_removed_{view_name} to generated_outputs/png/{timestamp}/bg_removed_{view_name}.png")
                
                logger.info(f"Prepared {len(images_without_background)} views for multi-view generation")
                
            else:
                # Fallback to original grid-based approach
                logger.info("Using legacy grid-based multi-view generation")
                
                # 2a. Generate multi-view grid using Qwen MV Edit
                grid_image = self.qwen_edit.edit_image_mv(prompt_image=image_edited, seed=request.seed)
                
                # 2b. Crop grid into individual views
                cropped_images = crop_grid_image(grid_image, grid_size=(2, 2))
                logger.info(f"Cropped grid into {len(cropped_images)} views: {list(cropped_images.keys())}")
                
                # Scale cropped images to target size
                target_size = (self.settings.qwen_edit_width, self.settings.qwen_edit_height)
                for view_name, cropped_img in cropped_images.items():
                    if cropped_img.size != target_size:
                        logger.info(f"{view_name} view size {cropped_img.size} -> scaling to {target_size}")
                        cropped_images[view_name] = cropped_img.resize(target_size, Image.Resampling.LANCZOS)
                
                # 2c. Remove background from each cropped view
                images_without_background = []
                view_names = ["front", "side", "back", "three_quarter"]
                fallback_view = "three_quarter"
                
                for view_name in view_names:
                    if view_name in cropped_images:
                        bg_removed = self.rmbg.remove_background(cropped_images[view_name])
                        images_without_background.append(bg_removed)
                    else:
                        # Use fallback logic
                        fallback = None
                        if fallback_view in cropped_images and fallback_view not in view_names:
                            fallback = fallback_view
                        elif "front" in cropped_images:
                            fallback = "front"
                        elif cropped_images:
                            fallback = list(cropped_images.keys())[0]
                        
                        if fallback:
                            logger.warning(f"Missing {view_name} view, using {fallback} as fallback")
                            bg_removed = self.rmbg.remove_background(cropped_images[fallback])
                            images_without_background.append(bg_removed)
                        else:
                            raise ValueError(f"Missing {view_name} view and no fallback available")
            
            # Ensure we have enough views for multi-view generation
            min_views = 2  # Minimum views needed
            if len(images_without_background) < min_views:
                raise ValueError(f"Expected at least {min_views} views for multi-view generation, got {len(images_without_background)}")
            
            # 3. Generate the 3D model using multi-view pipeline
            # Choose between Trellis and ReconViaGen based on settings
            use_reconviagen = getattr(self.settings, 'use_reconviagen', False)
            
            if use_reconviagen and self.reconviagen.is_ready():
                logger.info("Using ReconViaGen for multi-view 3D generation")
                
                # Extract parameters for ReconViaGen
                params = request.trellis_params
                ss_guidance_strength = getattr(params, 'sparse_structure_cfg_strength', 7.5)
                ss_sampling_steps = getattr(params, 'sparse_structure_steps', 30)
                slat_guidance_strength = getattr(params, 'slat_cfg_strength', 3.0)
                slat_sampling_steps = getattr(params, 'slat_steps', 12)
                multiimage_algo = getattr(self.settings, 'reconviagen_multiimage_algo', 'multidiffusion')
                
                trellis_result = self.reconviagen.generate_multiview(
                    images=images_without_background,
                    seed=request.seed,
                    ss_guidance_strength=ss_guidance_strength,
                    ss_sampling_steps=ss_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    slat_sampling_steps=slat_sampling_steps,
                    multiimage_algo=multiimage_algo
                )
            else:
                logger.info("Using standard Trellis for multi-view 3D generation")
                trellis_result = self.trellis.generate(
                    TrellisRequest(
                        image=images_without_background[0],  # Keep for backward compat
                        images=images_without_background,  # Multi-view images
                        seed=request.seed,
                        params=request.trellis_params
                    )
                )
            
            # For saving, use the first image without background
            image_without_background = images_without_background[0]
        else:
            # Single-view flow (backward compatible)
            # 2. Remove background
            image_without_background = self.rmbg.remove_background(image_edited)
           
            # 3. Generate the 3D model
            # Choose between Trellis and ReconViaGen based on settings
            use_reconviagen = getattr(self.settings, 'use_reconviagen', False)
            
            if use_reconviagen and self.reconviagen.is_ready():
                logger.info("Using ReconViaGen for single-view 3D generation")
                trellis_result = self.reconviagen.generate(
                    TrellisRequest(
                        image=image_without_background,
                        seed=request.seed,
                        params=request.trellis_params
                    )
                )
            else:
                logger.info("Using standard Trellis for single-view 3D generation")
                trellis_result = self.trellis.generate(
                    TrellisRequest(
                        image=image_without_background,
                        seed=request.seed,
                        params=request.trellis_params
                    )
                )

        # Save generated files
        if self.settings.save_generated_files:
            save_files(trellis_result, image_edited, image_without_background)
        
        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64 if self.settings.send_generated_files else None,
            image_without_background_file_base64=image_without_background_base64 if self.settings.send_generated_files else None,
        )
        return response

