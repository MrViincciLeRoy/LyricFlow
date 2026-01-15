import logging
import os
import tempfile
from typing import Optional, Dict, Any
from difflib import SequenceMatcher
import requests

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    return ''.join(c.lower() for c in text if c.isalnum())

def calculate_similarity(str1: str, str2: str) -> float:
    return SequenceMatcher(None, normalize_text(str1), normalize_text(str2)).ratio()

def search_lrclib(title: str, artist: str, strict_matching: bool = True) -> Optional[Dict[str, Any]]:
    """Search LRCLIB API for lyrics."""
    try:
        url = "https://lrclib.net/api/search"
        params = {"track_name": title, "artist_name": artist}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        if not results:
            return None
        
        best_match = None
        best_score = 0.0
        threshold = 0.85 if strict_matching else 0.65
        
        for result in results:
            title_sim = calculate_similarity(title, result.get('trackName', ''))
            artist_sim = calculate_similarity(artist, result.get('artistName', ''))
            combined_score = (title_sim + artist_sim) / 2
            
            if combined_score > best_score and result.get('syncedLyrics'):
                best_score = combined_score
                best_match = result
        
        if best_match and best_score >= threshold:
            logger.info(f"✓ Found match in LRCLIB (similarity: {best_score:.2f})")
            return {
                'provider': 'lrclib',
                'synced_lyrics': best_match['syncedLyrics'],
                'plain_lyrics': best_match.get('plainLyrics', ''),
                'match_score': best_score
            }
        
        logger.warning("⚠️  No high-quality match found in LRCLIB")
        return None
        
    except Exception as e:
        logger.error(f"❌ LRCLIB search failed: {e}")
        return None

def download_audio_temp(url: str) -> Optional[str]:
    """Download audio to temporary file."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        suffix = '.mp3'
        if 'content-type' in response.headers:
            content_type = response.headers['content-type']
            if 'ogg' in content_type:
                suffix = '.ogg'
            elif 'webm' in content_type:
                suffix = '.webm'
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        logger.info(f"✓ Downloaded audio to {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"❌ Audio download failed: {e}")
        return None

def whisper_fallback(title: str, artist: str, audio_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Generate lyrics using Whisper transcription."""
    try:
        from lyricflow.core.whisper_handler import transcribe_audio
        
        if not audio_path:
            logger.error("❌ Whisper fallback requires audio_path parameter")
            return None
        
        logger.info("🎤 Falling back to Whisper transcription...")
        transcription = transcribe_audio(audio_path)
        
        if transcription and transcription.get('segments'):
            synced_lyrics = '\n'.join([
                f"[{int(seg['start']//60):02d}:{seg['start']%60:05.2f}] {seg['text'].strip()}"
                for seg in transcription['segments']
            ])
            
            return {
                'provider': 'whisper_fallback',
                'synced_lyrics': synced_lyrics,
                'plain_lyrics': transcription.get('text', ''),
                'confidence': 'generated'
            }
        
        return None
        
    except ImportError:
        logger.error("❌ Whisper fallback not available (install whisper module)")
        return None
    except Exception as e:
        logger.error(f"❌ Whisper transcription failed: {e}")
        return None

def fetch_with_fallback(
    title: str,
    artist: str,
    audio_path: Optional[str] = None,
    audio_url: Optional[str] = None,
    strict_matching: bool = True,
    use_whisper: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Fetch lyrics from LRCLIB first, fallback to Whisper if needed.
    Automatically cleans up temporary audio files.
    
    Args:
        title: Song title
        artist: Artist name
        audio_path: Path to existing audio file
        audio_url: URL to download audio from (creates temp file)
        strict_matching: Require high similarity for LRCLIB matches
        use_whisper: Enable Whisper fallback if LRCLIB fails
    """
    temp_file = None
    
    try:
        result = search_lrclib(title, artist, strict_matching)
        
        if result:
            return result
        
        if not use_whisper:
            logger.warning("⚠️  Whisper fallback not enabled")
            return None
        
        # Handle audio source
        if audio_url and not audio_path:
            temp_file = download_audio_temp(audio_url)
            if not temp_file:
                return None
            audio_path = temp_file
        
        if not audio_path:
            logger.warning("⚠️  Whisper fallback enabled but no audio source provided")
            return None
        
        return whisper_fallback(title, artist, audio_path)
    
    finally:
        # Always cleanup temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.info(f"🗑️  Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to cleanup temp file: {e}")
