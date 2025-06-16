"""
Video Processing Module for Tree of Thoughts (ToT)

This module provides video processing capabilities using vLLM and Qwen VL models
for visual reasoning tasks within the Tree of Thoughts framework.
"""

import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from qwen_vl_utils import process_vision_info
import os
from typing import List, Dict, Any, Tuple, Optional


class VideoProcessor:
    """Video processor for handling video inputs in ToT framework."""
    
    def __init__(self, api_key: str = "EMPTY", api_base: str = "http://localhost:8000/v1"):
        """
        Initialize VideoProcessor with vLLM API configuration.
        
        Args:
            api_key: OpenAI API key (default "EMPTY" for vLLM)
            api_base: API base URL for vLLM server
        """
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
    
    def prepare_message_for_vllm(self, content_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prepare video messages for vLLM processing.
        
        The frame extraction logic for videos in vLLM differs from that of qwen_vl_utils.
        Here, we utilize qwen_vl_utils to extract video frames, with the media_type of 
        the video explicitly set to video/jpeg. By doing so, vLLM will no longer attempt 
        to extract frames from the input base64-encoded images.
        
        Args:
            content_messages: List of message dictionaries containing video content
            
        Returns:
            Tuple of (processed_messages, video_kwargs)
        """
        vllm_messages, fps_list = [], []
        
        for message in content_messages:
            message_content_list = message["content"]
            if not isinstance(message_content_list, list):
                vllm_messages.append(message)
                continue

            new_content_list = []
            for part_message in message_content_list:
                if 'video' in part_message:
                    video_message = [{'content': [part_message]}]
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        video_message, return_video_kwargs=True
                    )
                    assert video_inputs is not None, "video_inputs should not be None"
                    video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    fps_list.extend(video_kwargs.get('fps', []))

                    # Encode frames as base64
                    base64_frames = []
                    for frame in video_input:
                        img = Image.fromarray(frame)
                        output_buffer = BytesIO()
                        img.save(output_buffer, format="jpeg")
                        byte_data = output_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data).decode("utf-8")
                        base64_frames.append(base64_str)

                    part_message = {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                    }
                new_content_list.append(part_message)
            message["content"] = new_content_list
            vllm_messages.append(message)
            
        return vllm_messages, {'fps': fps_list}
    
    def process_video_query(
        self, 
        video_url: str, 
        query: str,
        model: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        total_pixels: int = 20480 * 28 * 28,
        min_pixels: int = 16 * 28 * 2,
        fps: float = 3.0,
        system_prompt: str = "You are a helpful assistant."
    ) -> str:
        """
        Process a video with a text query using the vision-language model.
        
        Args:
            video_url: URL or path to the video
            query: Text query about the video
            model: Model name to use for inference
            total_pixels: Total pixels for video processing
            min_pixels: Minimum pixels for video processing
            fps: Frames per second for video processing
            system_prompt: System prompt for the model
            
        Returns:
            Model response as string
        """
        video_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {
                    "type": "video",
                    "video": video_url,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    'fps': fps
                }]
            },
        ]
        
        processed_messages, video_kwargs = self.prepare_message_for_vllm(video_messages)
        
        chat_response = self.client.chat.completions.create(
            model=model,
            messages=processed_messages,
            extra_body={
                "mm_processor_kwargs": video_kwargs
            }
        )
        
        return chat_response.choices[0].message.content
    
    def analyze_video_for_tot(
        self, 
        video_url: str, 
        reasoning_prompt: str,
        model: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    ) -> str:
        """
        Analyze video content specifically for Tree of Thoughts reasoning.
        
        Args:
            video_url: URL or path to the video
            reasoning_prompt: Prompt designed for ToT reasoning
            model: Model name to use for inference
            
        Returns:
            Analysis result for ToT processing
        """
        system_prompt = (
            "You are an expert visual analyst helping with systematic reasoning. "
            "Analyze the video content carefully and provide structured insights "
            "that can be used for step-by-step problem solving."
        )
        
        return self.process_video_query(
            video_url=video_url,
            query=reasoning_prompt,
            model=model,
            system_prompt=system_prompt
        )


def create_video_processor(api_key: Optional[str] = None, api_base: Optional[str] = None) -> VideoProcessor:
    """
    Factory function to create a VideoProcessor instance.
    
    Args:
        api_key: OpenAI API key (defaults to environment variable or "EMPTY")
        api_base: API base URL (defaults to environment variable or localhost)
        
    Returns:
        VideoProcessor instance
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    
    if api_base is None:
        api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    
    return VideoProcessor(api_key=api_key, api_base=api_base)


# Example usage function
def example_video_analysis():
    """Example of how to use the video processor."""
    processor = create_video_processor()
    
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    query = "请用表格总结一下视频中的商品特点"
    
    result = processor.process_video_query(video_url, query)
    print("Video Analysis Result:", result)
    
    return result


if __name__ == "__main__":
    example_video_analysis()
