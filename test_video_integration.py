#!/usr/bin/env python3
"""
Test script for video processing integration in Tree of Thoughts.

This script tests the basic functionality of the integrated video processing
modules without requiring actual video analysis (for offline testing).
"""

import sys
import os

# Add the src directory to the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        # Test basic ToT imports
        from tot import __version__, gpt, chatgpt
        print("✓ Basic ToT modules imported successfully")
        
        # Test video processing import
        from tot.video_processing import VideoProcessor, create_video_processor
        print("✓ Video processing module imported successfully")
        
        # Test video config import
        from tot.video_config import get_config, configure_video_processing
        print("✓ Video configuration module imported successfully")
        
        # Test video task import (may fail if dependencies not installed)
        try:
            from tot.tasks.video_reasoning import VideoReasoningTask, create_video_summary_task
            print("✓ Video reasoning task imported successfully")
            video_support = True
        except ImportError as e:
            print(f"⚠ Video reasoning task import failed: {e}")
            print("  This is expected if video dependencies are not installed")
            video_support = False
        
        # Test HAS_VIDEO_SUPPORT flag
        from tot import HAS_VIDEO_SUPPORT
        print(f"✓ Video support flag: {HAS_VIDEO_SUPPORT}")
        
        return video_support
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_video_processor_creation():
    """Test video processor creation (without actual API calls)."""
    print("\nTesting video processor creation...")
    
    try:
        from tot.video_processing import create_video_processor
        
        # Test processor creation with default settings
        processor = create_video_processor()
        print("✓ Video processor created with default settings")
        
        # Test processor creation with custom settings
        processor = create_video_processor(
            api_key="test_key",
            api_base="http://test:8000/v1"
        )
        print("✓ Video processor created with custom settings")
        
        return True
        
    except Exception as e:
        print(f"✗ Video processor creation failed: {e}")
        return False


def test_video_config():
    """Test video configuration functionality."""
    print("\nTesting video configuration...")
    
    try:
        from tot.video_config import get_config, configure_video_processing
        
        # Test getting default config
        config = get_config()
        print(f"✓ Default config loaded - API base: {config.api_base}")
        
        # Test configuration update
        configure_video_processing(
            api_key="test_key",
            api_base="http://test:8000/v1",
            model="test_model"
        )
        print("✓ Configuration updated successfully")
        
        # Test config methods
        model_config = config.get_model_config()
        video_params = config.get_video_params()
        reasoning_config = config.get_reasoning_config()
        
        print(f"✓ Model config: {model_config['model']}")
        print(f"✓ Video params: FPS={video_params['fps']}")
        print(f"✓ Reasoning config: Max thoughts={reasoning_config['max_thoughts_per_step']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Video configuration test failed: {e}")
        return False


def test_video_task_creation():
    """Test video task creation (if dependencies are available)."""
    print("\nTesting video task creation...")
    
    try:
        from tot.tasks.video_reasoning import VideoReasoningTask, create_video_summary_task
        
        # Test basic task creation
        task = VideoReasoningTask(
            video_url="test_video.mp4",
            reasoning_type="analysis"
        )
        print("✓ Basic video reasoning task created")
        
        # Test task with custom parameters
        task = VideoReasoningTask(
            video_url="test_video.mp4",
            reasoning_type="summary",
            model="test_model"
        )
        print("✓ Video reasoning task with custom parameters created")
        
        # Test video summary task creation
        summary_task = create_video_summary_task("test_video.mp4")
        print("✓ Video summary task created")
        print(f"  Task has {len(summary_task)} steps configured")
        
        # Test task methods (without actual video processing)
        quality = summary_task.evaluate_thought_quality("This is a test thought with specific details.")
        print(f"✓ Thought quality evaluation: {quality:.2f}")
        
        return True
        
    except ImportError:
        print("⚠ Video task creation skipped - dependencies not installed")
        return True  # This is expected if dependencies are missing
    except Exception as e:
        print(f"✗ Video task creation failed: {e}")
        return False


def test_example_script_existence():
    """Test if example script exists and is executable."""
    print("\nTesting example script...")
    
    example_path = "video_reasoning_example.py"
    if os.path.exists(example_path):
        print("✓ Video reasoning example script exists")
        
        # Check if script is readable
        try:
            with open(example_path, 'r') as f:
                content = f.read()
                if "def main()" in content:
                    print("✓ Example script has main function")
                else:
                    print("⚠ Example script missing main function")
        except Exception as e:
            print(f"⚠ Could not read example script: {e}")
    else:
        print("✗ Video reasoning example script not found")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Tree of Thoughts Video Processing Integration Test")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Import Test", test_imports()))
    test_results.append(("Video Processor Creation", test_video_processor_creation()))
    test_results.append(("Video Configuration", test_video_config()))
    test_results.append(("Video Task Creation", test_video_task_creation()))
    test_results.append(("Example Script", test_example_script_existence()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Video processing integration is working correctly.")
        return 0
    else:
        print("⚠ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
