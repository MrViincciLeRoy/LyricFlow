"""
lyricflow/core/whisper_fallback.py - Whisper ASR fallback for when LRCLIB fails

This module provides automatic lyrics generation using Whisper ASR
when online lyrics databases don't have exact matches.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

logger = logging.getLogger(__name__)


class WhisperFallbackGenerator:
    """Generate lyrics using Whisper ASR as fallback."""
    
    def __init__(
        self,
        model_size: str = "medium",
        download_dir: Optional[Path] = None,
        max_line_duration: float = 5.0
    ):
        """
        Initialize Whisper fallback generator.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            download_dir: Directory for downloaded audio files
            max_line_duration: Maximum duration per LRC line in seconds
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper is required for fallback. Install with: pip install openai-whisper"
            )
        
        self.model_size = model_size
        self.model = None
        self.download_dir = download_dir or Path("downloads")
        self.download_dir.mkdir(exist_ok=True)
        self.max_line_duration = max_line_duration
    
    def _load_model(self):
        """Lazy load Whisper model."""
        if self.model is None:
            logger.info(f"⏳ Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("✓ Whisper model loaded")
    
    def download_from_youtube(
        self,
        title: str,
        artist: Optional[str] = None
    ) -> Optional[str]:
        """
        Download song from YouTube.
        
        Args:
            title: Song title
            artist: Artist name
            
        Returns:
            Path to downloaded audio file or None
        """
        if not YTDLP_AVAILABLE:
            logger.error("yt-dlp not available. Install with: pip install yt-dlp")
            return None
        
        search_query = f"{artist} - {title}" if artist else title
        logger.info(f"🔍 Searching YouTube for: {search_query}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.download_dir / f'{search_query}.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'default_search': 'ytsearch1',
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{search_query}", download=True)
                
                if 'entries' in info:
                    info = info['entries'][0]
                
                filename = ydl.prepare_filename(info)
                filename = filename.replace('.webm', '.mp3').replace('.m4a', '.mp3')
                
                logger.info(f"✅ Downloaded: {filename}")
                return filename
        
        except Exception as e:
            logger.error(f"❌ YouTube download error: {e}")
            return None
    
    def generate_lrc_from_audio(
        self,
        audio_path: str,
        artist: str = "Unknown Artist",
        title: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate LRC file from audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            artist: Artist name
            title: Song title
            output_path: Optional output path for LRC file
            
        Returns:
            Path to generated LRC file or None
        """
        if not os.path.exists(audio_path):
            logger.error(f"❌ Audio file not found: {audio_path}")
            return None
        
        self._load_model()
        
        logger.info(f"🎵 Extracting lyrics from: {os.path.basename(audio_path)}")
        logger.info("✓ Transcribing with word-level timestamps...")
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                compression_ratio_threshold=1.35,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            if title is None:
                title = os.path.basename(audio_path).replace('.mp3', '').replace('.m4a', '')
            
            # Determine output path
            if output_path is None:
                lrc_path = audio_path.replace('.mp3', '_whisper.lrc').replace('.m4a', '_whisper.lrc')
            else:
                lrc_path = output_path
            
            logger.info(f"📝 Creating LRC file (max {self.max_line_duration}s per line)...")
            
            # Generate LRC content
            lrc_content = self._create_lrc_content(result, artist, title)
            
            # Write to file
            with open(lrc_path, 'w', encoding='utf-8') as f:
                f.write(lrc_content)
            
            logger.info(f"✅ LRC file saved: {lrc_path}")
            
            # Also save plain text
            txt_path = lrc_path.replace('.lrc', '_lyrics.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            logger.info(f"✅ Plain text saved: {txt_path}")
            
            # Preview
            logger.info(f"\n📝 LRC Preview:\n{lrc_content[:500]}...")
            
            return lrc_path
        
        except Exception as e:
            logger.error(f"❌ Whisper transcription error: {e}")
            return None
    
    def _create_lrc_content(
        self,
        whisper_result: Dict[str, Any],
        artist: str,
        title: str
    ) -> str:
        """Create LRC file content from Whisper result."""
        lines = []
        
        # Metadata
        lines.append(f"[ar:{artist}]")
        lines.append(f"[ti:{title}]")
        lines.append(f"[by:LyricFlow - Whisper ASR Fallback]")
        
        # Duration
        if whisper_result.get('segments'):
            total_duration = whisper_result['segments'][-1]['end']
            lines.append(f"[length:{int(total_duration // 60):02d}:{int(total_duration % 60):02d}]")
        
        lines.append("")  # Empty line after metadata
        
        # Process segments with word-level timestamps
        for segment in whisper_result['segments']:
            if 'words' not in segment or not segment['words']:
                continue
            
            current_line_words = []
            current_line_start = None
            
            for word_info in segment['words']:
                word = word_info['word'].strip()
                start = word_info['start']
                end = word_info['end']
                
                if current_line_start is None:
                    current_line_start = start
                    current_line_words = [word]
                elif (end - current_line_start) <= self.max_line_duration:
                    current_line_words.append(word)
                else:
                    # Write current line
                    minutes = int(current_line_start // 60)
                    seconds = current_line_start % 60
                    timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
                    line_text = ' '.join(current_line_words)
                    lines.append(f"{timestamp}{line_text}")
                    
                    # Start new line
                    current_line_start = start
                    current_line_words = [word]
            
            # Write remaining words
            if current_line_words:
                minutes = int(current_line_start // 60)
                seconds = current_line_start % 60
                timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
                line_text = ' '.join(current_line_words)
                lines.append(f"{timestamp}{line_text}")
        
        return '\n'.join(lines)
    
    def generate_fallback(
        self,
        title: str,
        artist: str,
        audio_path: Optional[str] = None,
        download_from_youtube: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Complete fallback workflow: download + transcribe.
        
        Args:
            title: Song title
            artist: Artist name
            audio_path: Existing audio file (if available)
            download_from_youtube: Whether to download from YouTube if no audio file
            
        Returns:
            Dictionary with lyrics data or None
        """
        logger.info(f"🔄 Starting Whisper fallback for: '{title}' by '{artist}'")
        
        # Step 1: Get audio file
        if audio_path and os.path.exists(audio_path):
            logger.info(f"✓ Using existing audio file: {audio_path}")
        elif download_from_youtube:
            logger.info("📥 No audio file provided, downloading from YouTube...")
            audio_path = self.download_from_youtube(title, artist)
            if not audio_path:
                logger.error("❌ Failed to download audio")
                return None
        else:
            logger.error("❌ No audio file provided and download disabled")
            return None
        
        # Step 2: Generate LRC
        lrc_path = self.generate_lrc_from_audio(audio_path, artist, title)
        
        if not lrc_path:
            return None
        
        # Step 3: Read generated LRC
        with open(lrc_path, 'r', encoding='utf-8') as f:
            lrc_content = f.read()
        
        # Return in same format as LRCLIB
        return {
            'provider': 'whisper_fallback',
            'title': title,
            'artist': artist,
            'album': '',
            'duration': 0,
            'synced_lyrics': lrc_content,
            'plain_lyrics': None,
            'translation': None,
            'romanization': None,
            'instrumental': False,
            'has_synced': True,
            'has_plain': False,
            'source': 'Whisper ASR'
        }


# Convenience function
def generate_lyrics_fallback(
    title: str,
    artist: str,
    audio_path: Optional[str] = None,
    model_size: str = "medium",
    download_from_youtube: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for generating fallback lyrics.
    
    Args:
        title: Song title
        artist: Artist name
        audio_path: Path to audio file (optional)
        model_size: Whisper model size
        download_from_youtube: Whether to download from YouTube
        
    Returns:
        Lyrics data dictionary or None
    """
    generator = WhisperFallbackGenerator(model_size=model_size)
    return generator.generate_fallback(title, artist, audio_path, download_from_youtube)
