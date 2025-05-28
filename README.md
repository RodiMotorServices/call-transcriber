# Call Transcriber 🎙️

A powerful call center transcription tool that converts MP3 audio files into structured text with speaker separation (agent vs. client). The output is specifically formatted for AI consumption and knowledge base integration.

## 🚀 Features

- **High-Quality Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **Multi-Language Support**: Auto-detects language or specify manually (Spanish, English, French, etc.)
- **Speaker Diarization**: Automatically separates agent and client speech
- **AI-Ready Output**: Structured JSON format optimized for AI processing
- **Multiple Model Sizes**: Support for different Whisper model sizes (tiny to large)
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Progress Tracking**: Real-time progress indicators with beautiful CLI interface
- **Batch Processing**: Process multiple files with intelligent handling of already processed files
- **Smart File Management**: Automatically move processed files to organized directories

## 📋 Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- At least 4GB RAM (8GB+ recommended for larger models)

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## 🛠️ Installation

1. **Clone or download this project:**
```bash
git clone <repository-url>
cd call-transcriber
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Quick Testing Installation (Ubuntu/Debian):**
```bash
python3 quick_install.py
```

**Test Installation:**
```bash
python3 test_installation.py
```

## 🏗️ Project Structure

```
call-transcriber/
├── main.py                    # Core transcription engine
├── batch_process.py           # Batch processing with smart file handling
├── config.py                  # Configuration settings and optimizations
├── requirements.txt           # Python dependencies
├── quick_install.py          # Quick setup script for testing
├── test_installation.py      # Installation verification
├── setup.py                  # Package setup configuration
├── sample_output.json        # Example output format
├── mp3/                      # Default directory for audio files
│   ├── *.mp3                # Audio files to process
│   ├── *_transcription.json # Generated transcriptions
│   └── processed/           # Moved files after processing
└── README.md
```

## 📖 Usage

### Single File Processing

```bash
# Auto-detect language
python3 main.py path/to/your/call.mp3

# Specify Spanish language
python3 main.py path/to/your/call.mp3 --language es

# Specify English language with preview
python3 main.py path/to/your/call.mp3 --language en --preview

# Use larger model for better accuracy
python3 main.py call.mp3 --model medium --language es
```

### Batch Processing (New Enhanced Features)

```bash
# Process all MP3 files in directory (skips already processed)
python3 batch_process.py --directory ./mp3

# Process and move completed files to 'processed' subdirectory
python3 batch_process.py --directory ./mp3 --move-processed

# Force reprocess all files (ignore existing transcriptions)
python3 batch_process.py --directory ./mp3 --force-reprocess

# Process specific files
python3 batch_process.py --files call1.mp3 call2.mp3

# Use specific model and language for batch processing
python3 batch_process.py --directory ./mp3 --model medium --language es --move-processed

# Sequential processing (instead of parallel)
python3 batch_process.py --directory ./mp3 --sequential --move-processed
```

### Command Line Options

**main.py (Single File Processing):**
- `MP3_FILE`: Path to the MP3 audio file to transcribe (required)
- `--output, -o`: Output JSON file path (default: `{filename}_transcription.json`)
- `--model, -m`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) - default: `base`
- `--device, -d`: Processing device (`auto`, `cpu`, `cuda`, `mps`) - default: `auto`
- `--language, -l`: Language code or `auto` for detection - default: `auto`
- `--preview, -p`: Show results preview in terminal

**batch_process.py (Batch Processing):**
- `--directory, -d`: Directory containing MP3 files
- `--files, -f`: Specific files to process (multiple)
- `--pattern, -p`: File pattern to match (default: `*.mp3`)
- `--output-dir, -o`: Output directory for transcriptions
- `--model, -m`: Whisper model size - default: `base`
- `--device`: Processing device - default: `auto`
- `--language, -l`: Language code - default: `auto`
- `--workers, -w`: Number of parallel workers - default: `2`
- `--sequential`: Process files sequentially instead of parallel
- `--report, -r`: Output file for batch report
- `--skip-processed`: Skip files already transcribed (default: `True`)
- `--force-reprocess`: Force reprocessing of all files
- `--move-processed`: Move processed files to "processed" subdirectory

### Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `auto` | Auto-detect | `es` | Spanish |
| `en` | English | `fr` | French |
| `de` | German | `pt` | Portuguese |
| `it` | Italian | `ja` | Japanese |
| `ko` | Korean | `zh` | Chinese |
| `ru` | Russian | `ar` | Arabic |

*And 90+ more languages supported by Whisper*

### Model Sizes Comparison

| Model | Size | Speed | Accuracy | Memory | Use Case |
|-------|------|--------|----------|---------|----------|
| `tiny` | 39 MB | Fastest | Good | ~1GB | Quick processing, testing |
| `base` | 74 MB | Fast | Better | ~1GB | Balanced performance (default) |
| `small` | 244 MB | Medium | Good | ~2GB | Higher accuracy needs |
| `medium` | 769 MB | Slow | Very Good | ~5GB | Professional transcription |
| `large` | 1550 MB | Slowest | Excellent | ~10GB | Maximum accuracy requirements |

## 🎯 Smart Batch Processing Features

### Automatic Processed File Detection
The batch processor automatically:
- Detects already transcribed files (looks for `*_transcription.json`)
- Skips processed files by default (use `--force-reprocess` to override)
- Shows which files are being skipped

### Organized File Management
- `--move-processed`: Automatically moves completed audio and transcription files to a `processed/` subdirectory
- Keeps your main directory clean with only unprocessed files
- Maintains file associations (audio + transcription moved together)

### Processing Status Tracking
- Real-time progress bars with file-by-file status
- Comprehensive batch reports with statistics
- Success/failure tracking per file
- Processing time and speed metrics

## 📊 Output Format

The tool generates a comprehensive JSON file with the following structure:

```json
{
  "metadata": {
    "transcription_date": "2024-01-15T10:30:00.123456",
    "total_duration": 125.7,
    "language": "es",
    "agent_talk_time": 67.2,
    "client_talk_time": 58.5,
    "agent_talk_percentage": 53.5,
    "client_talk_percentage": 46.5
  },
  "conversation_flow": [
    {
      "speaker": "AGENT",
      "timestamp": "0.00s - 3.50s",
      "text": "Gracias por llamar a soporte técnico, mi nombre es María. ¿En qué puedo ayudarle?"
    },
    {
      "speaker": "CLIENT", 
      "timestamp": "4.20s - 8.10s",
      "text": "Hola María, tengo problemas para iniciar sesión en mi cuenta."
    }
  ],
  "agent_segments": {
    "full_text": "Complete agent speech...",
    "segments": [...],
    "segment_count": 15
  },
  "client_segments": {
    "full_text": "Complete client speech...",
    "segments": [...],
    "segment_count": 12
  },
  "ai_ready_format": {
    "call_summary": {
      "agent_speech": "Consolidated agent text for AI processing",
      "client_speech": "Consolidated client text for AI processing",
      "key_metrics": {
        "total_duration_seconds": 125.7,
        "agent_dominance_ratio": 0.53,
        "conversation_turns": 27
      }
    }
  }
}
```

## 🔧 Configuration

### Customizable Settings in `config.py`

- **Speaker Settings**: Pause thresholds, speaker labels, language-specific optimizations
- **Audio Processing**: Sample rates, quality settings, telephony optimizations
- **Whisper Settings**: Model configurations, temperature, beam size
- **Output Formats**: JSON formatting, metadata inclusion
- **Performance**: Batch processing, memory management, GPU acceleration

### Language-Specific Optimizations

```python
# Spanish configuration example
SPANISH_CONFIG = {
    "recommended_model": "medium",
    "pause_threshold": 1.8,
    "speaker_labels": {"agent": "AGENTE", "client": "CLIENTE"}
}
```

## 🤖 AI Integration

The output is specifically designed for AI consumption:

### For Knowledge Bases
- **Separated Content**: Agent and client speech clearly separated
- **Metadata Rich**: Comprehensive metrics for call analysis
- **Structured Format**: Easy to parse and index
- **Language-Aware**: Proper handling of accents and regional dialects

### Usage Examples

```python
import json

# Load transcription
with open('call_transcription.json', 'r') as f:
    data = json.load(f)

# Extract for AI processing
agent_text = data['ai_ready_format']['call_summary']['agent_speech']
client_text = data['ai_ready_format']['call_summary']['client_speech']
language = data['metadata']['language']

# Process conversation flow
for turn in data['conversation_flow']:
    print(f"{turn['speaker']} ({language}): {turn['text']}")
```

## 📈 Batch Processing Examples

### Basic Workflow
```bash
# 1. Process all new files in mp3 directory
python3 batch_process.py --directory ./mp3 --language es

# 2. Move processed files to keep directory organized
python3 batch_process.py --directory ./mp3 --move-processed --language es

# 3. Generate detailed report
python3 batch_process.py --directory ./mp3 --report batch_results.json
```

### Production Workflow
```bash
# High-quality batch processing with file organization
python3 batch_process.py \
  --directory ./mp3 \
  --model medium \
  --language es \
  --move-processed \
  --workers 4 \
  --report production_$(date +%Y%m%d).json
```

## 🎯 Call Center Specific Features

- **Agent First Detection**: Assumes agent speaks first (customizable in config)
- **Professional Language Processing**: Optimized for business conversations
- **Metrics for Quality Assurance**: Talk time ratios, turn counts, etc.
- **Compliance Ready**: Structured format for regulatory requirements
- **Multi-Language Support**: Handle international call centers
- **File Organization**: Keep processed and unprocessed files separated

## 🐛 Troubleshooting

### Common Issues

1. **Poor Recognition**: Use `--model medium` or `large` with specific `--language`
2. **FFmpeg not found**: Install FFmpeg and ensure it's in your PATH
3. **CUDA out of memory**: Use smaller model or `--device cpu`
4. **Files not being skipped**: Check if transcription JSON files exist in same directory
5. **Permission errors when moving files**: Ensure write permissions in target directory

### File Management Issues

```bash
# Check what files would be processed
python3 batch_process.py --directory ./mp3 --skip-processed

# Force reprocess everything
python3 batch_process.py --directory ./mp3 --force-reprocess

# Only move already processed files without reprocessing
python3 batch_process.py --directory ./mp3 --move-processed
```

## 📄 Dependencies

The project uses the following core dependencies (see `requirements.txt`):

- `openai-whisper>=20231117` - Speech transcription
- `pyannote.audio>=3.1.1` - Speaker diarization
- `pydub>=0.25.1` - Audio processing
- `torch>=2.0.0` - Deep learning framework
- `torchaudio>=2.0.0` - Audio processing for PyTorch
- `click>=8.1.0` - Command line interface
- `rich>=13.7.0` - Beautiful terminal output
- `librosa>=0.10.1` - Audio analysis
- `numpy>=1.24.0` - Numerical computing

## 📄 License

This project is provided as-is for call center transcription needs. Please ensure compliance with your local privacy and recording laws.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements to enhance the call transcription capabilities.

---

**Ready to transform your call center audio into AI-ready insights with smart file management!** 🚀 