"""
ReconViaGen Pipeline Integration for Single Image Input
Full ReconViaGen VGGT implementation for single image 3D generation
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
    print("✅ ReconViaGen imports successful - using local ReconViaGen VGGT implementation")
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


class ReconViaGenSingleImagePipeline:
    """
    ReconViaGen VGGT pipeline wrapper for single image input
    """
    
    def __init__(self, model_path: str = "Stable-X/trellis-vggt-v0-2"):
        self.model_path = model_path
        self.pipeline = None
        self.is_loaded = False
        self.has_vggt_model = False  # Track if we have the full VGGT model
        
    def load_pipeline(self) -> bool:
        """Load the ReconViaGen VGGT pipeline"""
        if not RECONVIAGEN_AVAILABLE:
            print("ReconViaGen dependencies not available")
            return False
            
        try:
            print(f"Loading ReconViaGen VGGT pipeline from {self.model_path}...")
            self.pipeline = TrellisVGGTTo3DPipeline.from_pretrained(self.model_path)
            self.pipeline.cuda()
            
            # Check if we have the full VGGT model
            # These attributes are available after loading the pretrained model
            if hasattr(self.pipeline, 'VGGT_model') and self.pipeline.VGGT_model is not None:
                if hasattr(self.pipeline, 'birefnet_model') and self.pipeline.birefnet_model is not None:
                    # Move models to CUDA
                    self.pipeline.VGGT_model.cuda()
                    self.pipeline.birefnet_model.cuda()
                    self.has_vggt_model = True
                    print("✅ Full ReconViaGen VGGT pipeline loaded")
                    print(f"   - VGGT model: {type(self.pipeline.VGGT_model).__name__}")
                    print(f"   - BiRefNet model: {type(self.pipeline.birefnet_model).__name__}")
                else:
                    self.has_vggt_model = False
                    print("⚠️ VGGT model found but BiRefNet model missing")
            else:
                self.has_vggt_model = False
                print("⚠️ VGGT model not found - using basic pipeline")
            
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
        Generate 3D model from single image using ReconViaGen VGGT pipeline
        
        Args:
            image: Input PIL Image
            seed: Random seed
            ss_guidance_strength: Sparse structure guidance strength
            ss_sampling_steps: Sparse structure sampling steps
            slat_guidance_strength: SLat guidance strength
            slat_sampling_steps: SLat sampling steps
            preprocess_image: Whether to preprocess the image
            
        Returns:
            PLY file as bytes or None if failed
        """
        if not self.is_loaded:
            if not self.load_pipeline():
                raise RuntimeError("Failed to load ReconViaGen VGGT pipeline")
        
        try:
            print(f"Generating 3D model from single image using ReconViaGen VGGT...")
            
            # Validate image
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {image.size}")
            
            # Ensure image is RGBA for ReconViaGen
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Generate 3D model using ReconViaGen VGGT pipeline (single image)
            # VGGT pipeline expects list[Image.Image] or torch.Tensor, so wrap single image in a list
            outputs = self.pipeline.run(
                image=[image],  # Wrap single image in list for VGGT pipeline
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
            import traceback
            traceback.print_exc()
            return None
    
    @property
    def device(self):
        """Get the device of the pipeline"""
        if self.pipeline is not None:
            return next(self.pipeline.parameters()).device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

