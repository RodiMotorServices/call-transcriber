# Call Transcriber 🎙️

A powerful call center transcription tool that converts MP3 audio files into structured text with speaker separation (agent vs. client). The output is specifically formatted for AI consumption and knowledge base integration.

## 🚀 Features

- **High-Quality Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **Multi-Language Support**: Auto-detects language or specify manually (Spanish, English, French, etc.)
- **Speaker Diarization**: Automatically separates agent and client speech
- **AI-Ready Output**: Structured JSON format optimized for AI processing
- **🤖 AI-Powered Categorization**: Automatic call categorization using Google Gemini API
- **📊 Intelligent Analysis**: Extract insights, topics, and action items from transcriptions
- **🔧 Centralized Configuration**: Environment-based API key management with robust error handling
- **Multiple Model Sizes**: Support for different Whisper model sizes (tiny to large)
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Progress Tracking**: Real-time progress indicators with beautiful CLI interface
- **Batch Processing**: Process multiple files with intelligent handling of already processed files
- **Smart File Management**: Automatically move processed files to organized directories
- **🔄 Retry Logic**: Built-in retry mechanisms for robust API interactions

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

4. **Environment Configuration:**
Create a `.env` file in the project root for API keys:
```bash
# Google Gemini API (for AI categorization)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: HuggingFace token for advanced speaker diarization
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Optional: OpenAI API key (for future integrations)
OPENAI_API_KEY=your_openai_api_key_here
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
├── main.py                         # Core transcription engine
├── batch_process.py                # Batch processing with smart file handling
├── categorize_transcriptions.py    # 🤖 AI-powered call categorization
├── summarize_transcriptions.py     # AI transcription summarization
├── config.py                       # Centralized configuration with API management
├── .env                            # 🔑 Environment variables (API keys)
├── requirements.txt                # Python dependencies
├── quick_install.py               # Quick setup script for testing
├── test_installation.py           # Installation verification
├── setup.py                       # Package setup configuration
├── sample_output.json             # Example output format
├── mp3/                           # Default directory for audio files
│   ├── *.mp3                      # Audio files to process
│   ├── *_transcription.json       # Generated transcriptions
│   ├── categories/                # 📊 AI categorization results
│   ├── categorized/               # Processed transcription files
│   └── processed/                 # Moved files after processing
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

### 🤖 AI-Powered Categorization (NEW!)

After transcription, automatically categorize calls using Google Gemini API:

```bash
# Categorize all transcribed files (basic usage)
python3 categorize_transcriptions.py -d ./mp3/transcriptions

# Process only 5 files for testing
python3 categorize_transcriptions.py -d ./mp3/transcriptions -n 5

# Use custom prompt template
python3 categorize_transcriptions.py -d ./mp3/transcriptions -p custom_prompt.txt

# Skip already categorized files (default behavior)
python3 categorize_transcriptions.py -d ./mp3/transcriptions --skip-processed

# Specify custom output directory
python3 categorize_transcriptions.py -d ./mp3/transcriptions -o ./results/categories
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

**categorize_transcriptions.py (🤖 AI Categorization):**
- `--directory, -d`: Directory containing transcription JSON files (required)
- `--output-dir, -o`: Output directory for category files (default: `../categories/`)
- `--prompt-file, -p`: Custom prompt template file
- `--skip-processed`: Skip already categorized files (default: `True`)
- `--limit, -n`: Limit number of files to process (useful for testing)

**🔑 Note**: Categorization requires `GEMINI_API_KEY` in your `.env` file

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

### Centralized Configuration in `config.py`

The project features a centralized configuration system that manages all settings:

- **🔑 Environment Variables**: Automatic loading from `.env` file
- **🤖 Gemini API Settings**: Model selection, generation parameters, safety settings
- **Speaker Settings**: Pause thresholds, speaker labels, language-specific optimizations
- **Audio Processing**: Sample rates, quality settings, telephony optimizations
- **Whisper Settings**: Model configurations, temperature, beam size
- **Output Formats**: JSON formatting, metadata inclusion
- **Performance**: Batch processing, memory management, GPU acceleration
- **🔄 Retry Logic**: Configurable retry attempts and delays for API calls

### API Configuration

```python
# Automatic environment loading
from config import get_gemini_client, validate_gemini_config

# Validate API configuration
is_valid, message = validate_gemini_config()

# Get configured Gemini client
model = get_gemini_client()
```

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

### 📊 Transcription Output
The output is specifically designed for AI consumption:

#### For Knowledge Bases
- **Separated Content**: Agent and client speech clearly separated
- **Metadata Rich**: Comprehensive metrics for call analysis
- **Structured Format**: Easy to parse and index
- **Language-Aware**: Proper handling of accents and regional dialects

#### Usage Examples

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

### 🤖 AI-Powered Categorization

The categorization system uses Google Gemini API to analyze transcriptions and extract structured insights:

#### Default Categories (Spanish Call Center)
- **"Quiere ir al taller"**: Customer wants to schedule service
- **"Cambio de cita"**: Appointment changes (advance, delay, cancel)
- **"Coche en taller"**: Vehicle status inquiries  
- **"Otros"**: General information, billing, complaints

#### Example Categorization Output

```json
{
  "categoria_principal": "Quiere ir al taller",
  "detalles": {
    "tipologia_reparacion": "Neumáticos",
    "terminos_clave": ["cambio de neumáticos", "Falken"],
    "menciones_marca_neumaticos": ["Falken"],
    "es_llamada_comercial": "NO",
    "presupuesto": "SI",
    "cita_concertada": "SI"
  }
}
```

#### Custom Prompts
Create custom analysis prompts for different business needs:

```python
# Custom prompt example
custom_prompt = """
Analyze this call and categorize based on:
1. Customer satisfaction level
2. Issue complexity  
3. Resolution status

Agent: {agent_speech}
Customer: {client_speech}
"""
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

#### Transcription Issues
1. **Poor Recognition**: Use `--model medium` or `large` with specific `--language`
2. **FFmpeg not found**: Install FFmpeg and ensure it's in your PATH
3. **CUDA out of memory**: Use smaller model or `--device cpu`
4. **Files not being skipped**: Check if transcription JSON files exist in same directory
5. **Permission errors when moving files**: Ensure write permissions in target directory

#### 🤖 AI Categorization Issues
6. **"Gemini API key not found"**: Ensure `GEMINI_API_KEY` is set in your `.env` file
7. **API rate limiting**: Use `--limit` parameter to process fewer files at once
8. **JSON parsing errors**: Check that your custom prompts return valid JSON format
9. **"Failed to initialize Gemini client"**: Verify your API key is valid and active

### File Management Issues

```bash
# Check what files would be processed
python3 batch_process.py --directory ./mp3 --skip-processed

# Force reprocess everything
python3 batch_process.py --directory ./mp3 --force-reprocess

# Only move already processed files without reprocessing
python3 batch_process.py --directory ./mp3 --move-processed
```

## 🚀 Complete Workflow

### Step-by-Step Process

1. **Setup Environment**
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
echo "GEMINI_API_KEY=your_key_here" > .env
```

2. **Transcribe Audio Files**
```bash
# Process all audio files
python3 batch_process.py --directory ./mp3 --language es --move-processed
```

3. **🤖 Categorize with AI**
```bash
# Analyze and categorize transcriptions
python3 categorize_transcriptions.py -d ./mp3/transcriptions -n 10
```

4. **Review Results**
```bash
# Check transcription results
ls mp3/processed/

# Check categorization results  
ls mp3/categories/
```

### Production Workflow Example

```bash
#!/bin/bash
# Complete call center processing pipeline

# 1. High-quality transcription
python3 batch_process.py \
  --directory ./mp3 \
  --model medium \
  --language es \
  --move-processed \
  --workers 4

# 2. AI-powered categorization
python3 categorize_transcriptions.py \
  -d ./mp3/transcriptions \
  --skip-processed

# 3. Generate reports
echo "Transcription complete: $(ls mp3/processed/*.json | wc -l) files"
echo "Categories generated: $(ls mp3/categories/*.json | wc -l) files"
```

## 📄 Dependencies

The project uses the following core dependencies (see `requirements.txt`):

### Core Transcription
- `openai-whisper>=20231117` - Speech transcription
- `pyannote.audio>=3.1.1` - Speaker diarization
- `pydub>=0.25.1` - Audio processing
- `torch>=2.0.0` - Deep learning framework
- `torchaudio>=2.0.0` - Audio processing for PyTorch

### 🤖 AI Integration
- `google-generativeai>=0.3.0` - Google Gemini API for AI categorization
- `python-dotenv>=1.0.0` - Environment variable management

### User Interface & Utilities
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