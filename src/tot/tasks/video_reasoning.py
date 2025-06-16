"""
Tree of Thoughts 框架的基于视频的推理任务

该模块扩展了基础任务类，以处理视频输入并在 Tree of Thoughts 方法论中进行视觉推理。
"""

from typing import List, Dict, Any, Optional
from .base import Task
from ..video_processing import create_video_processor, VideoProcessor


class VideoReasoningTask(Task):
    """
    使用视觉内容进行 Tree of Thoughts 推理的视频推理任务
    
    该任务扩展了基础 Task 类，用于处理视频输入并对视觉内容进行系统化推理。
    """
    
    def __init__(
        self, 
        video_url: str,
        reasoning_type: str = "analysis",
        model: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        初始化视频推理任务
        
        Args:
            video_url: 视频的 URL 或路径
            reasoning_type: 要执行的推理类型 (analysis, comparison 等)
            model: 要使用的视觉语言模型
            api_key: OpenAI API 密钥
            api_base: API 基础 URL
        """
        super().__init__()
        self.video_url = video_url
        self.reasoning_type = reasoning_type
        self.model = model
        self.video_processor = create_video_processor(api_key, api_base)
        self.steps = []
        self.current_step = 0
    
    def __len__(self) -> int:
        """返回推理步骤的数量"""
        return len(self.steps)
    
    def get_input(self, idx: int) -> str:
        """
        获取特定推理步骤的输入
        
        Args:
            idx: 步骤索引
            
        Returns:
            该步骤的输入提示
        """
        if idx < len(self.steps):
            return self.steps[idx].get("input", "")
        return ""
    
    def test_output(self, idx: int, output: str) -> Dict[str, Any]:
        """
        测试和评估推理步骤的输出
        
        Args:
            idx: 步骤索引
            output: 要评估的模型输出
            
        Returns:
            评估结果
        """
        # 基础评估 - 可以针对特定的视频推理任务进行扩展
        evaluation = {
            "step": idx,
            "output": output,
            "valid": len(output.strip()) > 0,
            "confidence": 0.5  # 默认置信度
        }
        
        # 如果有步骤特定的评估标准，则添加
        if idx < len(self.steps):
            step_info = self.steps[idx]
            if "evaluation_criteria" in step_info:
                # 应用自定义评估标准
                evaluation.update(self._evaluate_with_criteria(output, step_info["evaluation_criteria"]))
        
        return evaluation
    
    def _evaluate_with_criteria(self, output: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用特定标准评估输出
        
        Args:
            output: 模型输出
            criteria: 评估标准
            
        Returns:
            评估结果
        """
        results = {}
        
        # Check for required elements
        if "required_elements" in criteria:
            required = criteria["required_elements"]
            found_elements = sum(1 for element in required if element.lower() in output.lower())
            results["completeness"] = found_elements / len(required) if required else 1.0
        
        # Check output length
        if "min_length" in criteria:
            results["length_adequate"] = len(output) >= criteria["min_length"]
        
        # Check for structured format (e.g., table, list)
        if "format_type" in criteria:
            format_type = criteria["format_type"]
            if format_type == "table":
                results["has_table_format"] = "|" in output or "表格" in output
            elif format_type == "list":
                results["has_list_format"] = any(marker in output for marker in ["1.", "2.", "•", "-"])
        
        return results
    
    def setup_reasoning_steps(self, reasoning_prompts: List[Dict[str, Any]]) -> None:
        """
        Set up the reasoning steps for the video task.
        
        Args:
            reasoning_prompts: List of prompts and configurations for each reasoning step
        """
        self.steps = reasoning_prompts
        self.current_step = 0
    
    def analyze_video_content(self, prompt: str) -> str:
        """
        Analyze video content with a specific prompt.
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            Analysis result
        """
        return self.video_processor.analyze_video_for_tot(
            video_url=self.video_url,
            reasoning_prompt=prompt,
            model=self.model
        )
    
    def generate_initial_thoughts(self, base_prompt: str) -> List[str]:
        """
        Generate initial thoughts for Tree of Thoughts reasoning.
        
        Args:
            base_prompt: Base prompt for generating thoughts
            
        Returns:
            List of initial thoughts
        """
        thoughts = []
        
        # Generate different perspectives on the video
        perspectives = [
            f"{base_prompt} Focus on visual elements and composition.",
            f"{base_prompt} Focus on temporal changes and sequences.",
            f"{base_prompt} Focus on objects and their relationships.",
            f"{base_prompt} Focus on actions and behaviors."
        ]
        
        for perspective in perspectives:
            try:
                thought = self.analyze_video_content(perspective)
                thoughts.append(thought)
            except Exception as e:
                print(f"Error generating thought: {e}")
                thoughts.append(f"Unable to analyze: {perspective}")
        
        return thoughts
    
    def expand_thought(self, thought: str, expansion_prompt: str) -> List[str]:
        """
        Expand a thought into more detailed sub-thoughts.
        
        Args:
            thought: Original thought to expand
            expansion_prompt: Prompt for expansion
            
        Returns:
            List of expanded thoughts
        """
        full_prompt = f"""
        Based on this analysis of the video: {thought}
        
        {expansion_prompt}
        
        Provide detailed insights building on the previous analysis.
        """
        
        try:
            expansion = self.analyze_video_content(full_prompt)
            # Split expansion into multiple thoughts if it contains clear separators
            if "\n\n" in expansion:
                return [part.strip() for part in expansion.split("\n\n") if part.strip()]
            else:
                return [expansion]
        except Exception as e:
            print(f"Error expanding thought: {e}")
            return [f"Expansion failed: {thought}"]
    
    def evaluate_thought_quality(self, thought: str) -> float:
        """
        Evaluate the quality of a thought.
        
        Args:
            thought: Thought to evaluate
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Simple heuristic evaluation
        score = 0.0
        
        # Length check
        if len(thought) > 50:
            score += 0.3
        
        # Specificity check (contains specific details)
        specific_indicators = ["具体", "详细", "明确", "数据", "特征", "特点"]
        if any(indicator in thought for indicator in specific_indicators):
            score += 0.3
        
        # Structure check (contains organized information)
        structure_indicators = ["|", "表格", "列表", "1.", "2.", "3."]
        if any(indicator in thought for indicator in structure_indicators):
            score += 0.4
        
        return min(score, 1.0)


# Example usage
def create_video_summary_task(video_url: str) -> VideoReasoningTask:
    """
    Create a video summarization task.
    
    Args:
        video_url: URL of the video to analyze
        
    Returns:
        Configured VideoReasoningTask
    """
    task = VideoReasoningTask(video_url, reasoning_type="summary")
    
    # Set up reasoning steps for video summarization
    steps = [
        {
            "input": "请详细分析视频中的主要内容和关键信息。",
            "evaluation_criteria": {
                "required_elements": ["内容", "信息", "分析"],
                "min_length": 100,
                "format_type": "detailed"
            }
        },
        {
            "input": "基于前面的分析，请用表格形式总结视频中的商品特点。",
            "evaluation_criteria": {
                "required_elements": ["表格", "商品", "特点"],
                "min_length": 150,
                "format_type": "table"
            }
        },
        {
            "input": "请提供一个简洁的视频内容摘要，突出最重要的信息。",
            "evaluation_criteria": {
                "required_elements": ["摘要", "重要", "信息"],
                "min_length": 80,
                "format_type": "summary"
            }
        }
    ]
    
    task.setup_reasoning_steps(steps)
    return task


if __name__ == "__main__":
    # Example usage
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    task = create_video_summary_task(video_url)
    
    # Generate initial thoughts
    initial_thoughts = task.generate_initial_thoughts("请分析这个视频的内容。")
    print("Initial thoughts generated:", len(initial_thoughts))
    
    for i, thought in enumerate(initial_thoughts):
        print(f"Thought {i+1}: {thought[:100]}...")
