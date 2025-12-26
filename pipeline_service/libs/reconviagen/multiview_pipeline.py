"""
ReconViaGen Pipeline Integration for Multi-View Input
Full ReconViaGen implementation with multi-view support
"""

from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import os
import sys

# ReconViaGen imports - use local copy in libs directory
try:
    # Add the local ReconViaGen trellis path to sys.path
    libs_path = os.path.join(os.path.dirname(__file__), "..")
    
    if libs_path not in sys.path:
        sys.path.insert(0, libs_path)
    
    print(f"🔍 Importing ReconViaGen from local libs: {libs_path}")
    
    # Import from the local ReconViaGen trellis copy (now renamed to trellis)
    from trellis.pipelines.trellis_image_to_3d import TrellisVGGTTo3DPipeline
    from trellis.representations import Gaussian, MeshExtractResult
    from trellis.utils import render_utils, postprocessing_utils
    RECONVIAGEN_AVAILABLE = True
    print("✅ ReconViaGen imports successful - using local ReconViaGen implementation")
except ImportError as e:
    print(f"⚠️ Local ReconViaGen not available, falling back to basic trellis: {e}")
    # Fallback to basic trellis implementation
    libs_path = os.path.join(os.path.dirname(__file__), "..")
    if libs_path not in sys.path:
        sys.path.append(libs_path)
    
    try:
        from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline as TrellisVGGTTo3DPipeline
        from trellis.representations import Gaussian
        RECONVIAGEN_AVAILABLE = True
        print("✅ Using basic trellis implementation")
    except ImportError as e2:
        print(f"❌ ReconViaGen not available: {e2}")
        RECONVIAGEN_AVAILABLE = False
        TrellisVGGTTo3DPipeline = None
        Gaussian = None


class ReconViaGenMultiViewPipeline:
    """
    ReconViaGen pipeline wrapper for multi-view input
    """
    
    def __init__(self, model_path: str = "Stable-X/trellis-vggt-v0-1"):
        self.model_path = model_path
        self.pipeline = None
        self.is_loaded = False
        self.has_vggt_model = False  # Track if we have the full VGGT model
        
    def load_pipeline(self) -> bool:
        """Load the ReconViaGen pipeline"""
        if not RECONVIAGEN_AVAILABLE:
            print("ReconViaGen dependencies not available")
            return False
            
        try:
            print(f"Loading ReconViaGen pipeline from {self.model_path}...")
            self.pipeline = TrellisVGGTTo3DPipeline.from_pretrained(self.model_path)
            self.pipeline.cuda()
            
            # Check if we have the full VGGT model with multi-view support
            # These attributes are available after loading the pretrained model
            if hasattr(self.pipeline, 'VGGT_model') and self.pipeline.VGGT_model is not None:
                if hasattr(self.pipeline, 'birefnet_model') and self.pipeline.birefnet_model is not None:
                    # Move models to CUDA
                    self.pipeline.VGGT_model.cuda()
                    self.pipeline.birefnet_model.cuda()
                    self.has_vggt_model = True
                    print("✅ Full ReconViaGen VGGT pipeline loaded with multi-view support")
                    print(f"   - VGGT model: {type(self.pipeline.VGGT_model).__name__}")
                    print(f"   - BiRefNet model: {type(self.pipeline.birefnet_model).__name__}")
                else:
                    self.has_vggt_model = False
                    print("⚠️ VGGT model found but BiRefNet model missing")
            else:
                self.has_vggt_model = False
                print("⚠️ VGGT model not found - using basic pipeline")
            
            # Always check for run_multi_image method as backup indicator
            if hasattr(self.pipeline, 'run_multi_image'):
                print("✅ Multi-image generation method available")
            else:
                print("⚠️ Multi-image generation method not available")
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Failed to load ReconViaGen VGGT pipeline: {e}")
            return False
    
    def unload_pipeline(self):
        """Unload the pipeline to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            torch.cuda.empty_cache()
            print("ReconViaGen VGGT pipeline unloaded")
    
    def generate_3d_from_multiview_images(
        self,
        images: List[Image.Image],
        seed: int = 42,
        ss_guidance_strength: float = 7.5,
        ss_sampling_steps: int = 30,
        slat_guidance_strength: float = 3.0,
        slat_sampling_steps: int = 12,
        multiimage_algo: str = "multidiffusion",
        preprocess_image: bool = True
    ) -> Optional[bytes]:
        """
        Generate 3D model from multiple view images using ReconViaGen
        
        Args:
            images: List of input PIL Images (multi-view)
            seed: Random seed
            ss_guidance_strength: Sparse structure guidance strength
            ss_sampling_steps: Sparse structure sampling steps
            slat_guidance_strength: SLat guidance strength
            slat_sampling_steps: SLat sampling steps
            multiimage_algo: Multi-image algorithm ("multidiffusion" or "stochastic")
            preprocess_image: Whether to preprocess the images
            
        Returns:
            PLY file as bytes or None if failed
        """
        if not self.is_loaded:
            if not self.load_pipeline():
                raise RuntimeError("Failed to load ReconViaGen pipeline")
        
        try:
            print(f"Generating 3D model from {len(images)} multi-view images using ReconViaGen...")
            
            # Validate and prepare images for ReconViaGen
            valid_images = []
            for i, img in enumerate(images):
                if img.size[0] > 0 and img.size[1] > 0:
                    # Ensure image is RGBA for ReconViaGen
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    valid_images.append(img)
                else:
                    print(f"⚠️ Skipping image {i+1}: empty dimensions ({img.size})")
            
            if len(valid_images) < 2:
                print(f"⚠️ Not enough valid images for multi-view generation: {len(valid_images)}")
                # Use the first valid image or create fallback
                if len(valid_images) >= 1:
                    valid_images = [valid_images[0]]
                else:
                    # Create a fallback image
                    fallback_img = Image.new('RGBA', (512, 512), color=(128, 128, 128, 255))
                    valid_images = [fallback_img]
            
            print(f"✅ Using {len(valid_images)} valid images for 3D generation")
            
            # Use ReconViaGen multi-view generation (based on ReconViaGen-2 app.py)
            if len(valid_images) > 1:
                print(f"🎯 Using multi-view ReconViaGen generation with {len(valid_images)} images")
                # Use multi-image generation like in ReconViaGen-2
                outputs = self.pipeline.run(
                    image=valid_images,  # Pass list of images directly
                    seed=seed,
                    formats=["gaussian"],  # Only generate Gaussian splats
                    preprocess_image=preprocess_image,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "cfg_strength": ss_guidance_strength,
                    },
                    slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "cfg_strength": slat_guidance_strength,
                    },
                    mode=multiimage_algo,  # This parameter is supported in VGGT pipeline
                )
            else:
                print("🔄 Using single-view ReconViaGen generation")
                # Single image approach
                primary_image = valid_images[0]
                outputs = self.pipeline.run(
                    image=primary_image,
                    seed=seed,
                    formats=["gaussian"],  # Only generate Gaussian splats
                    preprocess_image=preprocess_image,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "cfg_strength": ss_guidance_strength,
                    },
                    slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "cfg_strength": slat_guidance_strength,
                    },
                )
            
            # Extract Gaussian from ReconViaGen output
            # ReconViaGen returns a tuple: (results_dict, tensor, tensor)
            if isinstance(outputs, tuple) and len(outputs) > 0:
                results_dict = outputs[0]  # First element contains the actual results
                if isinstance(results_dict, dict) and 'gaussian' in results_dict:
                    gaussian_output = results_dict['gaussian']
                    if isinstance(gaussian_output, (list, tuple)) and len(gaussian_output) > 0:
                        gs = gaussian_output[0]  # Get the first Gaussian object
                    else:
                        gs = gaussian_output
                else:
                    raise ValueError(f"Expected results dict with 'gaussian' key, got: {type(results_dict)}")
            elif isinstance(outputs, dict) and 'gaussian' in outputs:
                # Fallback for direct dict format
                gaussian_output = outputs['gaussian']
                gs = gaussian_output[0] if isinstance(gaussian_output, (list, tuple)) else gaussian_output
            else:
                raise ValueError(f"Unexpected ReconViaGen output format: {type(outputs)}")
            
            print(f"✅ Successfully extracted Gaussian: {type(gs)}")
            
            # Apply opacity filtering to improve quality (inspired by trellis_lotto_server_mod.py)
            opacity_threshold = 0.005
            opacity_mask = gs._opacity.squeeze() > opacity_threshold
            
            if opacity_mask.sum() < gs._opacity.shape[0]:
                remaining_points = opacity_mask.sum().item()
                total_points = gs._opacity.shape[0]
                removal_percentage = (total_points - remaining_points) / total_points * 100
                
                print(f"🔍 Filtering splats: {total_points} -> {remaining_points} (removed {removal_percentage:.1f}%)")
                
                # Only apply filtering if it's reasonable
                if remaining_points >= 1000 and removal_percentage <= 90:
                    # Apply mask to all Gaussian splat parameters
                    gs._xyz = gs._xyz[opacity_mask]
                    gs._features_dc = gs._features_dc[opacity_mask]
                    gs._scaling = gs._scaling[opacity_mask]
                    gs._rotation = gs._rotation[opacity_mask]
                    gs._opacity = gs._opacity[opacity_mask]
                    
                    # Only filter _features_rest if it exists
                    if hasattr(gs, '_features_rest') and gs._features_rest is not None:
                        gs._features_rest = gs._features_rest[opacity_mask]
                    
                    print("✅ Applied opacity filtering successfully")
                else:
                    print("⚠️ Skipping opacity filtering (too aggressive)")
            
            # Generate PLY data
            ply_buffer = io.BytesIO()
            gs.save_ply(ply_buffer)
            ply_buffer.seek(0)
            ply_data = ply_buffer.getvalue()
            
            print(f"✅ Generated 3D model: {len(ply_data)} bytes PLY")
            return ply_data
            
        except Exception as e:
            print(f"Error generating 3D model with ReconViaGen: {e}")
            return None
    
    def generate_3d_from_single_image(
        self,
        image: Image.Image,
        seed: int = 42,
        ss_guidance_strength: float = 7.5,
        ss_sampling_steps: int = 30,
        slat_guidance_strength: float = 3.0,
        slat_sampling_steps: int = 12,
        preprocess_image: bool = True
    ) -> Optional[bytes]:
        """
        Generate 3D model from single image using ReconViaGen
        Fallback for single image input
        """
        return self.generate_3d_from_multiview_images(
            images=[image],
            seed=seed,
            ss_guidance_strength=ss_guidance_strength,
            ss_sampling_steps=ss_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            slat_sampling_steps=slat_sampling_steps,
            multiimage_algo="multidiffusion",
            preprocess_image=preprocess_image
        )
    
    def _compress_ply_data(self, ply_data: bytes) -> bytes:
        """Compress PLY data using gzip"""
        import gzip
        return gzip.compress(ply_data)
    
    @property
    def device(self):
        """Get the device of the pipeline"""
        if self.pipeline is not None:
            return next(self.pipeline.parameters()).device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
