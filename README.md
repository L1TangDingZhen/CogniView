# CogniView - Dual-Layer Cognitive Video Analysis

A video monitoring and understanding system based on Vision-Language Models (VLM), using a dual-layer cognitive architecture for continuous observation and deep analysis of video content.

## Repository Name Suggestions

- **CogniView** - Cognitive + View (recommended)
- **VisionMind** - Vision + Mind
- **DualCogniVLM** - Dual Cognitive VLM
- **ObserveReason** - Observation + Reasoning layers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation Layer                         │
│  VLM: Qwen2-VL-2B / InternVL2                               │
│  Role: Continuous video observation, frame-level description │
└─────────────────────────────────────────────────────────────┘
                              ↓
                      Event Database (SQLite)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Understanding Layer                       │
│  LLM: Qwen2.5-3B/7B                                         │
│  Role: Activity inference, causal analysis, pattern discovery│
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- CUDA 12.x
- GPU: RTX 4070 (12GB) or higher

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention 2 for acceleration
pip install flash-attn --no-build-isolation
```

## Project Structure

```
video_monitor/
├── config.py                 # Configuration
├── models/
│   ├── vlm_loader.py         # Vision model loader
│   └── llm_loader.py         # Reasoning model loader
├── database/
│   └── event_db.py           # SQLite event database
├── utils/
│   ├── video_processor.py    # Video processing
│   ├── state_tracker.py      # State tracking
│   ├── output_parser.py      # Output parsing
│   └── sampling_strategy.py  # Sampling strategies
├── observation_layer.py      # Observation layer core
├── observation_service.py    # Observation service (continuous)
├── understanding_layer.py    # Understanding layer core
├── scheduler.py              # Scheduled task runner
├── run_test.py               # Testing tool
└── output/                   # Output directory
```

## Quick Start

### 1. Test Observation Layer

```bash
python run_test.py
```

Menu options:
- `[1]` Single model test - Process video with one VLM
- `[2]` Multi-model comparison - Compare different VLMs
- `[3]` Multi-frame motion detection - Test continuous frame action recognition

### 2. Run Observation Service

```bash
# Process video file
python observation_service.py -v /path/to/video.mp4

# Use camera
python observation_service.py -c 0

# Specify model and sampling strategy
python observation_service.py -v video.mp4 -m qwen2-vl-2b -s time_scheduled
```

Sampling strategies:
- `high_quality` - 2s/frame, high precision
- `realtime` - 10s/frame, near real-time
- `time_scheduled` - 2s daytime / 30s nighttime
- `adaptive` - Dynamic adjustment based on motion

### 3. Use Understanding Layer

```bash
# List available data
python understanding_layer.py -l

# Infer activity type
python understanding_layer.py -v video.mp4 -i

# Ask questions about video
python understanding_layer.py -v video.mp4 -a "What is the person doing?"

# Causal analysis
python understanding_layer.py -v video.mp4 -c "Why did they start cooking?"

# View timeline
python understanding_layer.py -v video.mp4 -t

# Discover behavior patterns
python understanding_layer.py -v video.mp4 -p

# Generate full report
python understanding_layer.py -v video.mp4 -r
```

### 4. Scheduled Analysis

```bash
# Start scheduler
python scheduler.py

# Run analysis immediately
python scheduler.py -r
```

## Supported Models

### Observation Layer (VLM)

| Model | VRAM | Speed | Notes |
|-------|------|-------|-------|
| qwen2-vl-2b | ~4GB | 2-4s/frame | Recommended, balanced |
| qwen2-vl-7b | ~8GB | - | Requires more VRAM |
| internvl2-1b | ~2GB | 1-2s/frame | Lightweight, Jetson-friendly |
| internvl2-2b | ~4GB | 2-3s/frame | Detailed descriptions |

### Understanding Layer (LLM)

| Model | VRAM | Notes |
|-------|------|-------|
| qwen2.5-1.5b | ~2GB | Fast, lightweight |
| qwen2.5-3b | ~4GB | Recommended, balanced |
| qwen2.5-7b | ~8GB | Strong reasoning |

## Core Features

### Observation Layer

- **Continuous Observation**: Extract frames from video/camera and generate descriptions
- **Multiple Sampling Strategies**: Fixed interval, time-scheduled, adaptive
- **State Tracking**: Track action duration and state changes
- **Structured Output**: Parse persons, actions, objects, scenes

### Understanding Layer

- **Activity Inference**: Infer activities from multiple observations
- **Causal Analysis**: Analyze cause-effect relationships between events
- **Pattern Discovery**: Find frequency, sequence, and temporal patterns
- **Free Q&A**: Natural language questions about video content
- **Report Generation**: Automatic analysis reports

## API Usage

```python
from understanding_layer import UnderstandingLayer

# Create understanding layer (load reasoning model)
layer = UnderstandingLayer(load_llm=True, llm_model="qwen2.5-3b")

# Infer activity
result = layer.infer_activity(
    video_name="dancing.mp4",
    start_time=0,
    end_time=30
)
print(f"Inferred activity: {result.inferred_activity}")
print(f"Confidence: {result.confidence}")

# Causal analysis
analysis = layer.analyze_causality(
    video_name="cooking.mp4",
    query="Why did they start chopping vegetables?"
)
print(f"Conclusion: {analysis.conclusion}")

# Free question
answer = layer.ask_question(
    video_name="video.mp4",
    question="How many people appear in the video?"
)
print(answer)

# Close (release GPU memory)
layer.close()
```

## Configuration

Edit `config.py` to customize:

```python
# Video directory
VIDEO_DIR = Path("../test_videos")

# Output directory
OUTPUT_DIR = Path("./output")

# Database path
DB_PATH = Path("./output/events.db")

# Default sampling interval
DEFAULT_SAMPLE_INTERVAL = 2.0
```

## Deployment on Jetson

For Jetson Orin Nano (8GB):

1. Use lightweight models:
   - Observation: `internvl2-1b` (1.77GB)
   - Understanding: `qwen2.5-1.5b` (2GB)

2. Use `realtime` or `time_scheduled` sampling strategy

3. Schedule understanding layer analysis during observation layer idle time

## Notes

- Observation and understanding layers use different models; manage VRAM carefully
- Avoid running both layers simultaneously (VRAM constraints)
- Use scheduled tasks to run understanding analysis during observation downtime
- GPU memory is automatically released when processing ends

## License

MIT
