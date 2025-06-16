#!/usr/bin/env python3
"""
Example script demonstrating video reasoning with Tree of Thoughts.

This script shows how to use the integrated video processing capabilities
within the Tree of Thoughts framework for visual reasoning tasks.
"""

import argparse
import sys
import os

# Add the src directory to the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from tot import HAS_VIDEO_SUPPORT
    if not HAS_VIDEO_SUPPORT:
        print("Video support is not available. Please install required dependencies:")
        print("pip install Pillow qwen-vl-utils 'openai>=1.0.0'")
        sys.exit(1)
    
    from tot.tasks.video_reasoning import create_video_summary_task, VideoReasoningTask
    from tot.video_processing import create_video_processor
except ImportError as e:
    print(f"Error importing video modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def basic_video_analysis(video_url: str, query: str = None):
    """
    Perform basic video analysis using the video processor.
    
    Args:
        video_url: URL or path to the video
        query: Custom query for the video (optional)
    """
    print("=== Basic Video Analysis ===")
    print(f"Video URL: {video_url}")
    
    # Create video processor
    processor = create_video_processor()
    
    # Use default query if none provided
    if query is None:
        query = "请用表格总结一下视频中的商品特点"
    
    print(f"Query: {query}")
    print("\nProcessing video...")
    
    try:
        result = processor.process_video_query(video_url, query)
        print("\n=== Analysis Result ===")
        print(result)
        return result
    except Exception as e:
        print(f"Error during video analysis: {e}")
        return None


def tree_of_thoughts_video_reasoning(video_url: str):
    """
    Perform video reasoning using Tree of Thoughts methodology.
    
    Args:
        video_url: URL or path to the video
    """
    print("\n=== Tree of Thoughts Video Reasoning ===")
    print(f"Video URL: {video_url}")
    
    # Create video reasoning task
    task = create_video_summary_task(video_url)
    
    print("\nGenerating initial thoughts...")
    
    # Generate initial thoughts from different perspectives
    base_prompt = "请分析这个视频的内容"
    initial_thoughts = task.generate_initial_thoughts(base_prompt)
    
    print(f"Generated {len(initial_thoughts)} initial thoughts:")
    for i, thought in enumerate(initial_thoughts, 1):
        print(f"\n--- Thought {i} ---")
        print(thought[:200] + "..." if len(thought) > 200 else thought)
        
        # Evaluate thought quality
        quality = task.evaluate_thought_quality(thought)
        print(f"Quality Score: {quality:.2f}")
    
    # Select best thoughts for expansion
    thought_qualities = [(i, task.evaluate_thought_quality(thought)) 
                        for i, thought in enumerate(initial_thoughts)]
    best_thoughts = sorted(thought_qualities, key=lambda x: x[1], reverse=True)[:2]
    
    print(f"\nExpanding top {len(best_thoughts)} thoughts...")
    
    expanded_thoughts = []
    for thought_idx, quality in best_thoughts:
        print(f"\n--- Expanding Thought {thought_idx + 1} (Quality: {quality:.2f}) ---")
        original_thought = initial_thoughts[thought_idx]
        
        expansion_prompt = "请基于这个分析，提供更详细的洞察和具体的观察结果。"
        expansions = task.expand_thought(original_thought, expansion_prompt)
        
        for j, expansion in enumerate(expansions):
            print(f"Expansion {j + 1}: {expansion[:150]}..." if len(expansion) > 150 else expansion)
            expanded_thoughts.append(expansion)
    
    # Final synthesis
    print("\n=== Final Analysis ===")
    synthesis_prompt = """
    基于以下所有的分析和洞察，请提供一个综合的视频内容总结，包括：
    1. 主要内容概述
    2. 关键特征和细节
    3. 结构化的信息表格
    
    之前的分析：
    """ + "\n\n".join(initial_thoughts + expanded_thoughts)
    
    try:
        final_analysis = task.analyze_video_content(synthesis_prompt)
        print(final_analysis)
        return final_analysis
    except Exception as e:
        print(f"Error during final synthesis: {e}")
        return None


def step_by_step_reasoning(video_url: str):
    """
    Perform step-by-step reasoning through predefined steps.
    
    Args:
        video_url: URL or path to the video
    """
    print("\n=== Step-by-Step Video Reasoning ===")
    
    task = create_video_summary_task(video_url)
    
    for step_idx in range(len(task)):
        print(f"\n--- Step {step_idx + 1} ---")
        step_input = task.get_input(step_idx)
        print(f"Prompt: {step_input}")
        
        try:
            result = task.analyze_video_content(step_input)
            print(f"Result: {result}")
            
            # Test the output
            evaluation = task.test_output(step_idx, result)
            print(f"Evaluation: {evaluation}")
            
        except Exception as e:
            print(f"Error in step {step_idx + 1}: {e}")


def main():
    """Main function for the video reasoning example."""
    parser = argparse.ArgumentParser(description="Video Reasoning with Tree of Thoughts")
    parser.add_argument("video_url", help="URL or path to the video to analyze")
    parser.add_argument("--query", help="Custom query for basic analysis")
    parser.add_argument("--mode", choices=["basic", "tot", "steps", "all"], 
                       default="all", help="Analysis mode to run")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--api-base", help="API base URL (or set OPENAI_API_BASE env var)")
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.api_base:
        os.environ["OPENAI_API_BASE"] = args.api_base
    
    print("Tree of Thoughts Video Reasoning Demo")
    print("=" * 40)
    
    try:
        if args.mode == "basic" or args.mode == "all":
            basic_video_analysis(args.video_url, args.query)
        
        if args.mode == "tot" or args.mode == "all":
            tree_of_thoughts_video_reasoning(args.video_url)
        
        if args.mode == "steps" or args.mode == "all":
            step_by_step_reasoning(args.video_url)
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
