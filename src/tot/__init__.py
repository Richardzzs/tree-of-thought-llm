__version__ = "0.1.0"

# Import core modules
from .models import gpt, chatgpt, gpt_usage
from .video_processing import VideoProcessor, create_video_processor

# Import tasks
from .tasks.base import Task
from .tasks.game24 import Game24Task
from .tasks.text import TextTask
from .tasks.crosswords import CrosswordsTask

# Try to import video reasoning task (requires additional dependencies)
try:
    from .tasks.video_reasoning import VideoReasoningTask, create_video_summary_task
    HAS_VIDEO_SUPPORT = True
except ImportError:
    HAS_VIDEO_SUPPORT = False
    VideoReasoningTask = None
    create_video_summary_task = None

# Import methods
from .methods.bfs import solve

__all__ = [
    'gpt', 'chatgpt', 'gpt_usage',
    'VideoProcessor', 'create_video_processor',
    'Task', 'Game24Task', 'TextTask', 'CrosswordsTask',
    'solve',
    'HAS_VIDEO_SUPPORT'
]

if HAS_VIDEO_SUPPORT:
    __all__.extend(['VideoReasoningTask', 'create_video_summary_task'])