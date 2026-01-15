"""
lyricflow/core/lrclib_with_fallback.py - LRCLIB with smart matching and Whisper fallback

This module enhances LRCLIB with:
1. Strict matching verification
2. Similarity scoring
3. Automatic Whisper ASR fallback for niche content
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from difflib import SequenceMatcher
import re

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedLRCLIBFetcher:
    """Enhanced LRCLIB fetcher with verification and fallback."""
    
    # Matching thresholds
    EXACT_MATCH_THRESHOLD = 0.95  # For exact matches
    GOOD_MATCH_THRESHOLD = 0.85   # For fuzzy matches
    MINIMUM_MATCH_THRESHOLD = 0.70  # Below this, use fallback
    
    BASE_URL = "https://lrclib.net/api"
    
    def __init__(
        self,
        use_whisper_fallback: bool = True,
        whisper_model: str = "medium",
        strict_matching: bool = True
    ):
        """
        Initialize enhanced LRCLIB fetcher.
        
        Args:
            use_whisper_fallback: Enable Whisper ASR fallback
            whisper_model: Whisper model size for fallback
            strict_matching: Require high similarity scores
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required")
        
        self.use_whisper_fallback = use_whisper_fallback
        self.whisper_model = whisper_model
        self.strict_matching = strict_matching
        self._whisper_generator = None
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()
    
    @staticmethod
    def similarity_score(a: str, b: str) -> float:
        """Calculate text similarity (0.0 to 1.0)."""
        a_norm = EnhancedLRCLIBFetcher.normalize_text(a)
        b_norm = EnhancedLRCLIBFetcher.normalize_text(b)
        
        if not a_norm or not b_norm:
            return 0.0
        
        return SequenceMatcher(None, a_norm, b_norm).ratio()
    
    def _get_whisper_generator(self):
        """Lazy load Whisper generator."""
        if self._whisper_generator is None:
            try:
                from lyricflow.core.whisper_fallback import WhisperFallbackGenerator
                self._whisper_generator = WhisperFallbackGenerator(
                    model_size=self.whisper_model
                )
                logger.info("✓ Whisper fallback initialized")
            except ImportError:
                logger.warning("⚠️  Whisper fallback not available")
                self.use_whisper_fallback = False
        
        return self._whisper_generator
    
    def fetch(
        self,
        title: str,
        artist: str,
        album: Optional[str] = None,
        duration: Optional[int] = None,
        audio_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch lyrics with smart matching and fallback.
        
        Args:
            title: Song title
            artist: Artist name  
            album: Album name (optional)
            duration: Duration in seconds (optional)
            audio_path: Path to audio file (for fallback)
            
        Returns:
            Lyrics data or None
        """
        logger.info(f"🔍 Fetching lyrics: '{title}' by '{artist}'")
        
        # Step 1: Try exact match via /get
        result = self._try_exact_match(title, artist, album, duration)
        
        if result and self._verify_match_quality(result, title, artist):
            logger.info("✅ High-quality exact match found")
            return result
        
        # Step 2: Try fuzzy search
        logger.info("🔍 Trying fuzzy search...")
        result = self._try_fuzzy_search(title, artist, album, duration)
        
        if result and self._verify_match_quality(result, title, artist):
            logger.info("✅ High-quality fuzzy match found")
            return result
        
        # Step 3: Fallback to Whisper if enabled
        if self.use_whisper_fallback:
            logger.warning("⚠️  No high-quality match found in LRCLIB")
            logger.info("🔄 Activating Whisper ASR fallback...")
            
            return self._whisper_fallback(title, artist, audio_path)
        else:
            logger.warning("❌ No suitable match found and fallback disabled")
            return None
    
    def _try_exact_match(
        self,
        title: str,
        artist: str,
        album: Optional[str],
        duration: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Try exact match via /get endpoint."""
        url = f"{self.BASE_URL}/get"
        params = {
            'artist_name': self._clean_text(artist),
            'track_name': self._clean_text(title),
        }
        
        if album:
            params['album_name'] = self._clean_text(album)
        if duration:
            params['duration'] = int(duration)
        
        try:
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_result(data)
            
        except Exception as e:
            logger.debug(f"Exact match failed: {e}")
        
        return None
    
    def _try_fuzzy_search(
        self,
        title: str,
        artist: str,
        album: Optional[str],
        duration: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Search and find best match."""
        url = f"{self.BASE_URL}/search"
        
        # Try multiple search strategies
        searches = [
            f"{artist} {title}",
            f"{title} {artist}",
            title
        ]
        
        all_results = []
        for query in searches:
            try:
                response = requests.get(url, params={'q': query}, timeout=5)
                if response.status_code == 200:
                    results = response.json()
                    all_results.extend(results)
            except Exception as e:
                logger.debug(f"Search '{query}' failed: {e}")
        
        if not all_results:
            return None
        
        # Score and rank
        scored = []
        for item in all_results:
            score = self._calculate_score(item, title, artist, album, duration)
            scored.append((score, item))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Log top matches
        logger.info(f"📊 Found {len(scored)} candidates:")
        for i, (score, item) in enumerate(scored[:3], 1):
            logger.info(f"  {i}. '{item.get('trackName')}' by '{item.get('artistName')}' - Score: {score:.2f}")
        
        # Return best if good enough
        best_score, best_item = scored[0]
        
        if best_score >= self.MINIMUM_MATCH_THRESHOLD:
            # Fetch full lyrics for best match
            return self._try_exact_match(
                best_item.get('trackName'),
                best_item.get('artistName'),
                best_item.get('albumName'),
                best_item.get('duration')
            )
        
        return None
    
    def _calculate_score(
        self,
        item: Dict[str, Any],
        title: str,
        artist: str,
        album: Optional[str],
        duration: Optional[int]
    ) -> float:
        """Calculate match score."""
        result_title = item.get('trackName', '')
        result_artist = item.get('artistName', '')
        
        # Title (50%)
        title_score = self.similarity_score(title, result_title) * 0.5
        
        # Artist (40%)
        artist_score = self.similarity_score(artist, result_artist) * 0.4
        
        # Album bonus (5%)
        album_score = 0.0
        if album and item.get('albumName'):
            album_score = self.similarity_score(album, item.get('albumName')) * 0.05
        
        # Duration bonus (5%)
        duration_score = 0.0
        if duration and item.get('duration'):
            diff = abs(duration - item.get('duration'))
            if diff <= duration * 0.05:  # 5% tolerance
                duration_score = 0.05
        
        return min(title_score + artist_score + album_score + duration_score, 1.0)
    
    def _verify_match_quality(
        self,
        result: Dict[str, Any],
        query_title: str,
        query_artist: str
    ) -> bool:
        """Verify if match quality is acceptable."""
        if not result:
            return False
        
        title_sim = self.similarity_score(query_title, result.get('title', ''))
        artist_sim = self.similarity_score(query_artist, result.get('artist', ''))
        
        logger.info(f"  📊 Match quality - Title: {title_sim:.2f}, Artist: {artist_sim:.2f}")
        
        if self.strict_matching:
            # Both must be high quality
            return (title_sim >= self.GOOD_MATCH_THRESHOLD and 
                    artist_sim >= self.GOOD_MATCH_THRESHOLD)
        else:
            # Average must be acceptable
            avg_sim = (title_sim + artist_sim) / 2
            return avg_sim >= self.MINIMUM_MATCH_THRESHOLD
    
    def _whisper_fallback(
        self,
        title: str,
        artist: str,
        audio_path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Use Whisper ASR as fallback."""
        generator = self._get_whisper_generator()
        
        if not generator:
            logger.error("❌ Whisper fallback not available")
            return None
        
        try:
            result = generator.generate_fallback(
                title=title,
                artist=artist,
                audio_path=audio_path,
                download_from_youtube=True
            )
            
            if result:
                logger.info("✅ Whisper fallback successful")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Whisper fallback failed: {e}")
            return None
    
    def _parse_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LRCLIB API response."""
        synced = data.get('syncedLyrics')
        plain = data.get('plainLyrics')
        
        # Prefer synced
        if synced and synced.strip():
            lyrics = synced
            is_synced = True
        elif plain:
            lyrics = plain
            is_synced = False
        else:
            return None
        
        return {
            'provider': 'lrclib',
            'id': data.get('id'),
            'title': data.get('trackName', ''),
            'artist': data.get('artistName', ''),
            'album': data.get('albumName', ''),
            'duration': data.get('duration', 0),
            'synced_lyrics': lyrics if is_synced else None,
            'plain_lyrics': lyrics if not is_synced else None,
            'translation': None,
            'romanization': None,
            'instrumental': data.get('instrumental', False),
            'has_synced': is_synced,
            'has_plain': not is_synced
        }
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for API query."""
        if not text:
            return ''
        text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
        return text.strip()
    
    def save_lrc(
        self,
        result: Dict[str, Any],
        output_path: Path
    ) -> bool:
        """Save lyrics to LRC file."""
        lyrics = result.get('synced_lyrics') or result.get('plain_lyrics')
        
        if not lyrics:
            logger.error("No lyrics to save")
            return False
        
        try:
            lrc_content = f"[ti:{result['title']}]\n"
            lrc_content += f"[ar:{result['artist']}]\n"
            
            if result.get('album'):
                lrc_content += f"[al:{result['album']}]\n"
            
            if result.get('duration'):
                dur = int(result['duration'])
                lrc_content += f"[length:{dur // 60:02d}:{dur % 60:02d}]\n"
            
            lrc_content += f"[by:LyricFlow - {result.get('provider', 'unknown')}]\n"
            lrc_content += "\n" + lyrics
            
            Path(output_path).write_text(lrc_content, encoding='utf-8')
            logger.info(f"✅ Saved to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False


# Convenience function
def fetch_with_fallback(
    title: str,
    artist: str,
    album: Optional[str] = None,
    audio_path: Optional[str] = None,
    strict_matching: bool = True,
    use_whisper: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Fetch lyrics with automatic Whisper fallback.
    
    Args:
        title: Song title
        artist: Artist name
        album: Album name
        audio_path: Path to audio file (for fallback)
        strict_matching: Require high similarity
        use_whisper: Enable Whisper fallback
        
    Returns:
        Lyrics data or None
    """
    fetcher = EnhancedLRCLIBFetcher(
        use_whisper_fallback=use_whisper,
        strict_matching=strict_matching
    )
    
    return fetcher.fetch(title, artist, album, audio_path=audio_path)
