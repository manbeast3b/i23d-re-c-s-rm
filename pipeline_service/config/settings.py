from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent

class Settings(BaseSettings):
    api_title: str = "3D Generation pipeline Service"

    # API settings
    host: str = "0.0.0.0"
    port: int = 10006

    # GPU settings
    qwen_gpu: int = Field(default=0, env="QWEN_GPU")
    trellis_gpu: int = Field(default=0, env="TRELLIS_GPU")
    dtype: str = Field(default="bf16", env="QWEN_DTYPE")

    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Generated files settings
    save_generated_files: bool = Field(default=False, env="SAVE_GENERATED_FILES")
    send_generated_files: bool = Field(default=False, env="SEND_GENERATED_FILES")
    output_dir: Path = Field(default=Path("generated_outputs"), env="OUTPUT_DIR")

    # Trellis settings
    trellis_model_id: str = Field(default="jetx/trellis-image-large", env="TRELLIS_MODEL_ID")
    trellis_sparse_structure_steps: int = Field(default=12, env="TRELLIS_SPARSE_STRUCTURE_STEPS")
    trellis_sparse_structure_cfg_strength: float = Field(default=5.75, env="TRELLIS_SPARSE_STRUCTURE_CFG_STRENGTH")
    trellis_slat_steps: int = Field(default=20, env="TRELLIS_SLAT_STEPS")
    trellis_slat_cfg_strength: float = Field(default=2.4, env="TRELLIS_SLAT_CFG_STRENGTH")
    trellis_num_oversamples: int = Field(default=3, env="TRELLIS_NUM_OVERSAMPLES")
    compression: bool = Field(default=False, env="COMPRESSION")
    
    # Multi-view Trellis settings (disabled for single-image pipeline)
    trellis_use_multi_view: bool = Field(default=False, env="TRELLIS_USE_MULTI_VIEW")
    trellis_sparse_structure_cfg_interval_start: float = Field(default=0.3, env="TRELLIS_SS_CFG_INTERVAL_START")
    trellis_sparse_structure_cfg_interval_end: float = Field(default=0.98, env="TRELLIS_SS_CFG_INTERVAL_END")
    trellis_sparse_structure_rescale_t: float = Field(default=3.0, env="TRELLIS_SS_RESCALE_T")
    trellis_slat_cfg_interval_start: float = Field(default=0.3, env="TRELLIS_SLAT_CFG_INTERVAL_START")
    trellis_slat_cfg_interval_end: float = Field(default=0.98, env="TRELLIS_SLAT_CFG_INTERVAL_END")
    trellis_slat_rescale_t: float = Field(default=3.0, env="TRELLIS_SLAT_RESCALE_T")
    trellis_mode: str = Field(default="stochastic", env="TRELLIS_MODE")

    # Qwen Edit settings
    qwen_edit_base_model_path: str = Field(default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",env="QWEN_EDIT_BASE_MODEL_PATH")
    qwen_edit_model_path: str = Field(default="Qwen/Qwen-Image-Edit-2509",env="QWEN_EDIT_MODEL_PATH")
    qwen_edit_height: int = Field(default=1024, env="QWEN_EDIT_HEIGHT")
    qwen_edit_width: int = Field(default=1024, env="QWEN_EDIT_WIDTH")
    num_inference_steps: int = Field(default=8, env="NUM_INFERENCE_STEPS")
    true_cfg_scale: float = Field(default=1.0, env="TRUE_CFG_SCALE")
    qwen_edit_prompt_path: Path = Field(default=config_dir.joinpath("qwen_edit_prompt.json"), env="QWEN_EDIT_PROMPT_PATH")
    qwen_mv_prompt_path: Path = Field(default=config_dir.joinpath("qwen_mv_prompt.json"), env="QWEN_MV_PROMPT_PATH")
    enable_first_block_cache: bool = Field(default=True, env="ENABLE_FIRST_BLOCK_CACHE")
    # Backgorund removal settings
    # background_removal_model_id: str = Field(default="hiepnd11/rm_back2.0", env="BACKGROUND_REMOVAL_MODEL_ID")
    background_removal_model_id: str = Field(default="mohantesting/remove_background", env="BACKGROUND_REMOVAL_MODEL_ID")
    input_image_size: tuple[int, int] = Field(default=(1024, 1024), env="INPUT_IMAGE_SIZE") # (height, width)
    output_image_size: tuple[int, int] = Field(default=(518, 518), env="OUTPUT_IMAGE_SIZE") # (height, width)
    padding_percentage: float = Field(default=0.2, env="PADDING_PERCENTAGE")
    limit_padding: bool = Field(default=True, env="LIMIT_PADDING")

    # ReconViaGen settings
    use_reconviagen: bool = Field(default=True, env="USE_RECONVIAGEN", description="Use ReconViaGen instead of standard Trellis")
    reconviagen_model_id: str = Field(default="Stable-X/trellis-vggt-v0-2", env="RECONVIAGEN_MODEL_ID")
    reconviagen_gpu: int = Field(default=0, env="RECONVIAGEN_GPU")
    reconviagen_multiimage_algo: str = Field(default="multidiffusion", env="RECONVIAGEN_MULTIIMAGE_ALGO", description="Multi-image algorithm: multidiffusion or stochastic")

    # Sequential multi-view generation settings (disabled for single-image pipeline)
    multiview_count: int = Field(default=0, env="MULTIVIEW_COUNT", description="Number of additional views to generate (disabled for single-image pipeline)")
    use_sequential_views: bool = Field(default=False, env="USE_SEQUENTIAL_VIEWS", description="Use sequential view generation (disabled for single-image pipeline)")
    qwen_view_prompts_path: Path = Field(default=config_dir.joinpath("qwen_view_prompts.json"), env="QWEN_VIEW_PROMPTS_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]

