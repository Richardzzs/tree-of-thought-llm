"""
Configuration for video processing in Tree of Thoughts framework.
"""

import os
from typing import Dict, Any, Optional


class VideoConfig:
    """Configuration class for video processing settings."""
    
    def __init__(self):
        """Initialize with default settings."""
        # API Configuration
        self.api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        self.api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
        
        # Model Configuration
        self.default_model = "Qwen/Qwen2.5-VL-32B-Instruct"
        self.fallback_models = [
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen-VL-Chat"
        ]
        
        # Video Processing Parameters
        self.default_total_pixels = 20480 * 28 * 28
        self.default_min_pixels = 16 * 28 * 2
        self.default_fps = 3.0
        
        # Reasoning Parameters
        self.max_thoughts_per_step = 4
        self.thought_expansion_limit = 3
        self.quality_threshold = 0.6
        
        # System Prompts
        self.system_prompts = {
            "general": "You are a helpful assistant.",
            "analysis": (
                "You are an expert visual analyst helping with systematic reasoning. "
                "Analyze the video content carefully and provide structured insights "
                "that can be used for step-by-step problem solving."
            ),
            "synthesis": (
                "You are an expert at synthesizing information from multiple sources. "
                "Combine the provided analyses into a coherent and comprehensive summary."
            )
        }
        
        # Evaluation Criteria Templates
        self.evaluation_templates = {
            "completeness": {
                "required_elements": [],
                "min_length": 50
            },
            "structure": {
                "format_type": "detailed",
                "organization_required": True
            },
            "accuracy": {
                "factual_consistency": True,
                "logical_flow": True
            }
        }
    
    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Args:
            model_name: Specific model name (optional)
            
        Returns:
            Model configuration dictionary
        """
        return {
            "model": model_name or self.default_model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "fallback_models": self.fallback_models
        }
    
    def get_video_params(self, **overrides) -> Dict[str, Any]:
        """
        Get video processing parameters.
        
        Args:
            **overrides: Parameter overrides
            
        Returns:
            Video parameters dictionary
        """
        params = {
            "total_pixels": self.default_total_pixels,
            "min_pixels": self.default_min_pixels,
            "fps": self.default_fps
        }
        params.update(overrides)
        return params
    
    def get_reasoning_config(self) -> Dict[str, Any]:
        """
        Get reasoning configuration.
        
        Returns:
            Reasoning configuration dictionary
        """
        return {
            "max_thoughts_per_step": self.max_thoughts_per_step,
            "thought_expansion_limit": self.thought_expansion_limit,
            "quality_threshold": self.quality_threshold
        }
    
    def get_system_prompt(self, prompt_type: str = "general") -> str:
        """
        Get system prompt by type.
        
        Args:
            prompt_type: Type of prompt to retrieve
            
        Returns:
            System prompt string
        """
        return self.system_prompts.get(prompt_type, self.system_prompts["general"])
    
    def get_evaluation_criteria(self, criteria_type: str = "completeness") -> Dict[str, Any]:
        """
        Get evaluation criteria template.
        
        Args:
            criteria_type: Type of criteria to retrieve
            
        Returns:
            Evaluation criteria dictionary
        """
        return self.evaluation_templates.get(criteria_type, self.evaluation_templates["completeness"]).copy()
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # API settings
        if "OPENAI_API_KEY" in os.environ:
            self.api_key = os.environ["OPENAI_API_KEY"]
        if "OPENAI_API_BASE" in os.environ:
            self.api_base = os.environ["OPENAI_API_BASE"]
        
        # Model settings
        if "TOT_VIDEO_MODEL" in os.environ:
            self.default_model = os.environ["TOT_VIDEO_MODEL"]
        
        # Video processing settings
        if "TOT_VIDEO_FPS" in os.environ:
            try:
                self.default_fps = float(os.environ["TOT_VIDEO_FPS"])
            except ValueError:
                pass
        
        if "TOT_MAX_THOUGHTS" in os.environ:
            try:
                self.max_thoughts_per_step = int(os.environ["TOT_MAX_THOUGHTS"])
            except ValueError:
                pass
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            filepath: Path to save the configuration
        """
        import json
        
        config_dict = {
            "api_key": self.api_key,
            "api_base": self.api_base,
            "default_model": self.default_model,
            "fallback_models": self.fallback_models,
            "video_params": {
                "total_pixels": self.default_total_pixels,
                "min_pixels": self.default_min_pixels,
                "fps": self.default_fps
            },
            "reasoning_params": {
                "max_thoughts_per_step": self.max_thoughts_per_step,
                "thought_expansion_limit": self.thought_expansion_limit,
                "quality_threshold": self.quality_threshold
            },
            "system_prompts": self.system_prompts,
            "evaluation_templates": self.evaluation_templates
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to load the configuration from
        """
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Update configuration
            self.api_key = config_dict.get("api_key", self.api_key)
            self.api_base = config_dict.get("api_base", self.api_base)
            self.default_model = config_dict.get("default_model", self.default_model)
            self.fallback_models = config_dict.get("fallback_models", self.fallback_models)
            
            if "video_params" in config_dict:
                vp = config_dict["video_params"]
                self.default_total_pixels = vp.get("total_pixels", self.default_total_pixels)
                self.default_min_pixels = vp.get("min_pixels", self.default_min_pixels)
                self.default_fps = vp.get("fps", self.default_fps)
            
            if "reasoning_params" in config_dict:
                rp = config_dict["reasoning_params"]
                self.max_thoughts_per_step = rp.get("max_thoughts_per_step", self.max_thoughts_per_step)
                self.thought_expansion_limit = rp.get("thought_expansion_limit", self.thought_expansion_limit)
                self.quality_threshold = rp.get("quality_threshold", self.quality_threshold)
            
            if "system_prompts" in config_dict:
                self.system_prompts.update(config_dict["system_prompts"])
            
            if "evaluation_templates" in config_dict:
                self.evaluation_templates.update(config_dict["evaluation_templates"])
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load configuration from {filepath}: {e}")


# Global configuration instance
video_config = VideoConfig()

# Update from environment on import
video_config.update_from_env()


def get_config() -> VideoConfig:
    """Get the global video configuration instance."""
    return video_config


def configure_video_processing(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> None:
    """
    Configure video processing settings.
    
    Args:
        api_key: OpenAI API key
        api_base: API base URL
        model: Default model name
        **kwargs: Additional configuration parameters
    """
    global video_config
    
    if api_key is not None:
        video_config.api_key = api_key
    if api_base is not None:
        video_config.api_base = api_base
    if model is not None:
        video_config.default_model = model
    
    # Update other parameters
    for key, value in kwargs.items():
        if hasattr(video_config, key):
            setattr(video_config, key, value)


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print("Current configuration:")
    print(f"API Base: {config.api_base}")
    print(f"Default Model: {config.default_model}")
    print(f"Default FPS: {config.default_fps}")
    
    # Save example configuration
    config.save_to_file("video_config_example.json")
    print("Example configuration saved to video_config_example.json")
