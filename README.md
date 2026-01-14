# LyricFlow

A modular Python toolkit for automated processing and embedding of song lyrics, featuring multi-provider fetching, powerful CLI, interactive TUI, and scalable FastAPI web service.

## ✨ Features

### Core Features
- 🎵 **Audio Metadata Handling**: Read and write lyrics tags in MP3, M4A, FLAC, OGG, and more
- � **Multi-Provider Lyrics Fetching**: Search from LRCLIB (free) or Musixmatch (commercial)
- 🖥️ **Three Interfaces**: CLI commands, Interactive TUI, and REST API
- �🇯🇵 **Japanese Romanization**: Local (pykakasi + fugashi) and AI-based (OpenAI/Gemini)
- 🔄 **Smart Workflow**: Check embedded lyrics → search .lrc files → fetch online → romanize → embed
- � **Translation Support**: Optional translation via OpenAI/Gemini/Musixmatch APIs
- 🎤 **ASR Generation**: Whisper-based automatic lyric generation
- ⚡ **Modular Design**: Each component is independently usable and replaceable

### New Multi-Provider System ✅
- **LRCLIB Provider**: Free, no API key required, synced lyrics support
- **Musixmatch Provider**: Large database, translations, romanization support
- **Unified Interface**: Switch providers seamlessly
- **Smart Fallback**: Automatic provider switching on failure
- **Interactive TUI**: Visual search and preview with Textual

## Installation

### Basic Installation

```bash
!pip install git+https://github.com/MrViincciLeRoy/LyricFlow.git#egg=LyricFlow[dev]
```

### With Optional Features

```bash
# Lyrics fetching (LRCLIB + Musixmatch)
pip install "git+https://github.com/MrViincciLeRoy/LyricFlow.git#egg=LyricFlow[fetch]"

# Interactive TUI
pip install "git+https://github.com/MrViincciLeRoy/LyricFlow.git#egg=LyricFlow[tui]"

# OpenAI support
pip install "git+https://github.com/MrViincciLeRoy/LyricFlow.git#egg=LyricFlow[openai]"

# Gemini support
pip install "git+https://github.com/MrViincciLeRoy/LyricFlow.git#egg=LyricFlow[gemini]"

# Whisper ASR
pip install "git+https://github.com/MrViincciLeRoy/LyricFlow.git#egg=LyricFlow[whisper]"

# FastAPI service
pip install "git+https://github.com/MrViincciLeRoy/LyricFlow.git#egg=LyricFlow[api]"

# Everything
pip install "git+https://github.com/MrViincciLeRoy/LyricFlow.git#egg=LyricFlow[all]"

```

### Development Installation

```bash
!git clone https://github.com/MrViincciLeRoy/LyricFlow.git && cd LyricFlow && pip install -e ".[dev]"

```

## Quick Start

### 🚀 Fetch Lyrics (New!)

#### Basic Fetch with CLI
```bash
# Fetch lyrics from LRCLIB (free, no API key needed)
lyricflow fetch -t "Yesterday" -a "The Beatles"

# Launch interactive TUI for visual search
lyricflow fetch --interactive

# Use Musixmatch provider (requires API key)
lyricflow fetch -t "Song Title" -a "Artist" --provider musixmatch

# Fetch with romanization
lyricflow fetch -t "Song Title" -a "Artist" --romanization

# Fetch with translation
lyricflow fetch -t "Song Title" -a "Artist" --translation --translate-to en

# Save to file
lyricflow fetch -t "Song Title" -a "Artist" -o lyrics.lrc

# Embed directly in audio file
lyricflow fetch -t "Song Title" -a "Artist" --embed audio.m4a
```

#### Interactive TUI
```bash
# Launch the beautiful Textual-based TUI
lyricflow fetch -i

# Inside TUI:
# 1. Choose provider (LRCLIB/Musixmatch)
# 2. Enter song details
# 3. Search and preview results
# 4. Save or embed lyrics
```

#### Provider Comparison
| Feature | LRCLIB | Musixmatch |
|---------|--------|------------|
| Cost | ✅ Free | ⚠️ API Key |
| Setup | ✅ None | ⚙️ Required |
| Synced Lyrics | ✅ Yes | ✅ Yes |
| Translations | ❌ No | ✅ Yes |
| Database | Medium | Very Large |

**Quick Test:**
```bash
# Test the system right now
python test_tui.py

# Run comprehensive demos
python demo_providers.py
```

### CLI Usage

#### Process a Single File

```bash
# Process with local romanization
lyricflow process song.mp3

# Process with AI romanization
lyricflow process song.mp3 --use-ai --api openai

# Process without embedding (just create .lrc files)
lyricflow process song.mp3 --no-embed
```

#### Process a Directory

```bash
# Process all audio files recursively
lyricflow process /path/to/music --recursive

# Process with overwrite
lyricflow process /path/to/music --overwrite
```

#### Romanize Text

```bash
# Romanize text directly
lyricflow romanize "こんにちは世界"

# Romanize from file
lyricflow romanize --file lyrics.txt

# Romanize with AI
lyricflow romanize "こんにちは世界" --use-ai --api gemini
```

#### View Configuration

```bash
lyricflow config
```

### Python API Usage

#### Fetch Lyrics
```python
from lyricflow.core.lyrics_provider import create_fetcher

# Create LRCLIB fetcher (free)
fetcher = create_fetcher("lrclib")

# Search for lyrics
result = fetcher.fetch("Yesterday", "The Beatles")

if result:
    print(f"Title: {result['title']}")
    print(f"Artist: {result['artist']}")
    
    # Get synced lyrics
    if result['synced_lyrics']:
        print(result['synced_lyrics'])
    
    # Save to file
    fetcher.save_lrc(result, "yesterday.lrc")
```

#### Process Audio Files
```python
from pathlib import Path
from lyricflow import LyricsSync, Romanizer, AudioHandler

# Process an audio file
lyrics_sync = LyricsSync()
result = lyrics_sync.process_audio_file(
    Path("song.mp3"),
```
    use_ai=False,
    overwrite=False
)
print(result)

# Romanize text
romanizer = Romanizer()
romanized = romanizer.romanize("こんにちは世界", language="ja")
print(romanized)  # "Konnichiwa sekai"

# Handle audio metadata
audio = AudioHandler(Path("song.mp3"))
metadata = audio.get_metadata()
print(f"Title: {metadata['title']}")
print(f"Artist: {metadata['artist']}")
```

### FastAPI Web Service

Start the server:

```bash
uvicorn lyricflow.api.server:app --reload
```

Or programmatically:

```python
from lyricflow.api.server import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

API endpoints:

- `GET /` - API information
- `POST /process` - Upload audio file for processing
- `GET /status/{task_id}` - Check processing status
- `POST /romanize` - Romanize text
- `GET /health` - Health check

Interactive API documentation available at: http://localhost:8000/docs

## Configuration

LyricFlow looks for configuration in the following order:

1. `./config.yaml` (project root)
2. `~/.lyricflow/config.yaml` (user home directory)
3. Environment variables

### Configuration File Example

```yaml
# config.yaml
api:
  default_provider: local  # local, openai, or gemini
  openai:
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"
  gemini:
    api_key: "..."

processing:
  language: auto
  translate_target: en
  skip_existing_lyrics: true
  on_failure: skip

whisper:
  model_size: medium
  device: cpu

caching:
  enabled: true
  ttl: 2592000
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export LYRICFLOW_API_PROVIDER="openai"
```

## Workflow

LyricFlow follows an efficient workflow to minimize processing:

1. **Check Embedded Lyrics**: If romanized lyrics already exist in the audio file tags, skip (unless `--overwrite`)
2. **Check Synced Lyrics**: If synced lyrics (SYLT tag) exist, romanize and embed them
3. **Search .lrc File**: Look for a matching `.lrc` file in the same directory
4. **Romanize & Embed**: Romanize the found lyrics and embed into audio file
5. **Generate Sidecar**: Create `{basename}_romaji.lrc` file
6. **Flag for ASR**: If no lyrics found, suggest using Whisper generation

## Features in Detail

### Local Romanization

Uses `pykakasi` and `fugashi` (MeCab) for high-quality Japanese romanization:

- Accurate segmentation with MeCab
- Proper Hepburn romanization
- Correct particle pronunciation (は→wa, を→o, へ→e)
- Long vowel handling (ō, ū, ē)
- Natural word spacing

### AI-Based Romanization

Supports OpenAI and Gemini APIs:

- More context-aware romanization
- Better handling of ambiguous text
- Custom base URL support for local LLMs
- Automatic fallback to local romanization

### Audio Format Support

Supported audio formats:
- MP3 (ID3v2 tags)
- M4A/MP4 (iTunes-style tags)
- FLAC (Vorbis comments)
- OGG (Vorbis comments)
- OPUS
- WMA

### Lyric Tag Types

- **SYLT**: Synced lyrics (ID3)
- **USLT**: Unsynced lyrics (ID3)
- **©lyr**: Lyrics (MP4)
- **TXXX:Romanized_Lyrics**: Custom romanized lyrics tag
- **----:com.apple.iTunes:Romanized_Lyrics**: Custom MP4 romanized lyrics

## CLI Reference

```bash
lyricflow --help

Commands:
  config     View current configuration
  generate   Generate lyrics from audio using Whisper ASR
  process    Process audio file(s) to romanize and embed lyrics
  romanize   Romanize Japanese text

Options:
  --version          Show version
  -v, --verbose      Enable verbose logging
  --help             Show help message
```

### Process Command

```bash
lyricflow process [OPTIONS] PATH

Options:
  --recursive / --no-recursive  Process directories recursively
  --api [local|openai|gemini]   Romanization API to use
  --use-ai                      Use AI romanization
  --overwrite                   Force reprocessing existing lyrics
  --no-embed                    Generate LRC files but do not embed
  --dry-run                     Simulate without making changes
```

## Development

### Project Structure

```
lyricflow/
├── lyricflow/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py          # FastAPI endpoints
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py            # Click CLI
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio_handler.py   # Audio metadata handling
│   │   ├── lrclib.py          # LRCLIB provider (NEW)
│   │   ├── musixmatch.py      # Musixmatch provider (NEW)
│   │   ├── lyrics_provider.py # Unified provider interface (NEW)
│   │   ├── lyrics_sync.py     # Main workflow orchestration
│   │   └── romanizer.py       # Romanization engine
│   ├── tui/
│   │   └── __init__.py        # Textual TUI (NEW)
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       └── logging.py         # Logging utilities
├── tests/
├── docs/
│   ├── PROVIDERS_GUIDE.md     # Provider comparison (NEW)
│   ├── MUSIXMATCH_GUIDE.md    # Musixmatch setup (NEW)
│   └── ...
├── demo_providers.py          # Interactive demos (NEW)
├── test_tui.py                # TUI test suite (NEW)
├── INTEGRATION_COMPLETE.md    # Complete summary (NEW)
├── QUICKSTART.md              # Quick reference (NEW)
├── config.yaml.example
├── .env.example
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lyricflow tests/

# Quick TUI test
python test_tui.py

# Interactive demos
python demo_providers.py
```

### Code Style

```bash
# Format code
black lyricflow tests

# Lint
ruff check lyricflow tests

# Type check
mypy lyricflow
```

## Roadmap

- [x] Local Japanese romanization (pykakasi + fugashi)
- [x] AI-based romanization (OpenAI/Gemini)
- [x] Audio metadata handling (mutagen)
- [x] CLI tool (click + rich)
- [x] FastAPI web service
- [x] Configuration management
- [x] **Multi-provider lyrics fetching (LRCLIB + Musixmatch)** ✨
- [x] **Interactive TUI with Textual** ✨
- [x] **Translation support (Musixmatch)** ✨
- [ ] Whisper ASR lyric generation
- [ ] Additional providers (Genius, AZLyrics)
- [ ] Caching layer for API calls
- [ ] WebSocket support for real-time progress
- [ ] Batch processing optimization
- [ ] Docker container
- [ ] Web UI

## 📚 Documentation

### Quick Links
- **[Quick Start Guide](QUICKSTART.md)** - Get started in 30 seconds
- **[Integration Complete](INTEGRATION_COMPLETE.md)** - Full feature summary and testing results
- **[Providers Guide](docs/PROVIDERS_GUIDE.md)** - Compare LRCLIB vs Musixmatch
- **[Musixmatch Setup](docs/MUSIXMATCH_GUIDE.md)** - How to set up Musixmatch API

### Demo Scripts
- **`demo_providers.py`** - Interactive demonstrations of all features
- **`test_tui.py`** - Automated test suite with 6 validation tests

### Status: ✅ Production Ready
All three interfaces (CLI, TUI, API) are fully functional with comprehensive testing.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [LRCLIB](https://lrclib.net/) - Free synced lyrics database
- [Musixmatch](https://www.musixmatch.com/) - Large commercial lyrics database
- [Textual](https://textual.textualize.io/) - Modern TUI framework
- [pykakasi](https://github.com/miurahr/pykakasi) - Japanese romanization
- [fugashi](https://github.com/polm/fugashi) - MeCab wrapper
- [mutagen](https://github.com/quodlibet/mutagen) - Audio metadata handling
- [click](https://click.palletsprojects.com/) - CLI framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [rich](https://rich.readthedocs.io/) - Terminal formatting

## Support

- 📧 Email: hello@sibikrish.dev
- 💬 Issues: https://github.com/Sibikrish3000/LyricFlow/issues
- 📖 Documentation: https://lyricflow.readthedocs.io

---

Made with ❤️ by the LyricFlow team
