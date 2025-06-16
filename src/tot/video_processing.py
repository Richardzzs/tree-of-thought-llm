"""
Tree of Thoughts (ToT) 视频处理模块

该模块使用 vLLM 和 Qwen VL 模型为 Tree of Thoughts 框架提供视频处理能力，
支持在视觉推理任务中进行系统化思考。
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
    """用于处理 ToT 框架中视频输入的处理器类"""
    
    def __init__(self, api_key: str = "EMPTY", api_base: str = "http://localhost:8000/v1"):
        """
        使用 vLLM API 配置初始化视频处理器
        
        Args:
            api_key: OpenAI API 密钥 (vLLM 默认使用 "EMPTY")
            api_base: vLLM 服务器的 API 基础 URL
        """
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
    
    def prepare_message_for_vllm(self, content_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        为 vLLM 处理准备视频消息
        
        vLLM 中视频的帧提取逻辑与 qwen_vl_utils 不同。
        这里我们使用 qwen_vl_utils 来提取视频帧，并将视频的 media_type 
        明确设置为 video/jpeg。这样 vLLM 就不会再尝试从输入的 base64 编码图像中提取帧。
        
        Args:
            content_messages: 包含视频内容的消息字典列表
            
        Returns:
            元组：(处理后的消息, 视频参数)
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
        处理视频和文本查询，使用视觉语言模型进行推理
        
        Args:
            video_url: 视频的 URL 或路径
            query: 关于视频的文本查询
            model: 用于推理的模型名称
            total_pixels: 视频处理的总像素
            min_pixels: 视频处理的最小像素
            fps: 视频处理的帧率
            system_prompt: 模型的系统提示
            
        Returns:
            模型响应的字符串
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
        针对 Tree of Thoughts 推理分析视频内容
        
        Args:
            video_url: 视频的 URL 或路径
            reasoning_prompt: 专为 ToT 推理设计的提示语
            model: 用于推理的模型名称
            
        Returns:
            ToT 处理的分析结果
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
    工厂函数，用于创建 VideoProcessor 实例
    
    Args:
        api_key: OpenAI API 密钥 (默认为环境变量或 "EMPTY")
        api_base: API 基础 URL (默认为环境变量或 localhost)
        
    Returns:
        VideoProcessor 实例
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    
    if api_base is None:
        api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    
    return VideoProcessor(api_key=api_key, api_base=api_base)


# Example usage function
def example_video_analysis():
    """使用视频处理器的示例"""
    processor = create_video_processor()
    
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    query = "请用表格总结一下视频中的商品特点"
    
    result = processor.process_video_query(video_url, query)
    print("Video Analysis Result:", result)
    
    return result


if __name__ == "__main__":
    example_video_analysis()
