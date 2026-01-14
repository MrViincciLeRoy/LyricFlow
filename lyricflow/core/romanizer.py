"""Romanization module for LyricFlow.

Supports both local (pykakasi + fugashi) and AI-based (OpenAI/Gemini) romanization.
"""

import re
from typing import Optional, Literal
from abc import ABC, abstractmethod

try:
    import pykakasi
    import fugashi
    LOCAL_ROMANIZATION_AVAILABLE = True
except ImportError:
    LOCAL_ROMANIZATION_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from lyricflow.utils.logging import get_logger
from lyricflow.utils.config import Config

logger = get_logger(__name__)


def clean_lrc_timestamps(text: str) -> str:
    """
    Clean LRC timestamps to proper format.
    
    Fixes:
    - Spaces inside brackets: [ 00 : 01 . 61 ] -> [00:01.61]
    - Spaces after brackets: [00:01.61] text -> [00:01.61]text
    
    Args:
        text: Text with potential timestamp issues
        
    Returns:
        Text with cleaned timestamps (preserves newlines)
    """
    # First, remove spaces inside timestamp brackets
    # Pattern: [ 00 : 01 . 61 ] -> [00:01.61]
    text = re.sub(r'\[\s*(\d+)\s*:\s*(\d+)\s*\.\s*(\d+)\s*\]', r'[\1:\2.\3]', text)
    
    # Then, remove ONLY horizontal spaces after timestamp (not newlines)
    # Pattern: [00:01.61] text -> [00:01.61]text
    # Use [ \t]+ to match only spaces and tabs, NOT newlines
    text = re.sub(r'(\[\d+:\d+\.\d+\])[ \t]+', r'\1', text)
    
    return text


class RomanizerBase(ABC):
    """Abstract base class for romanizers."""
    
    @abstractmethod
    def romanize(self, text: str, language: str = "ja") -> str:
        """Romanize the given text."""
        pass


class LocalRomanizer(RomanizerBase):
    """Local romanization using pykakasi and fugashi."""
    
    def __init__(self):
        if not LOCAL_ROMANIZATION_AVAILABLE:
            raise ImportError(
                "Local romanization requires pykakasi and fugashi. "
                "Install with: pip install pykakasi fugashi unidic-lite"
            )
        
        self.tagger = fugashi.Tagger()
        self.kks = pykakasi.kakasi()
        logger.info("Local romanizer initialized with pykakasi and fugashi")
    
    def post_process_romaji(self, text: str) -> str:
        """Apply post-processing rules for accurate Hepburn romanization."""
        # Long vowels - convert to macrons
        text = text.replace('oo', 'ō').replace('ou', 'ō')
        text = text.replace('uu', 'ū')
        text = text.replace('ei', 'ē')
        
        # No-macron exceptions for common words
        no_macron_words = {
            'unmē': 'unmei', 'sē': 'sei', 'ēen': 'eien',
            'mē': 'mei', 'kē': 'kei', 'rē': 'rei', 'tē': 'tei'
        }
        for with_macron, without_macron in no_macron_words.items():
            text = text.replace(with_macron, without_macron)
        
        # Common pronunciation fixes
        pronunciation_fixes = {
            'mabataki': 'matataki', 'bai o': 'hai o',
            'deha ': 'dewa ', 'niha ': 'niwa ',
            'he ': 'e ', 'wa kanai': 'hakanai',
            'maru de wa kanai': 'marude hakanai',
            'wa takushi': 'watakushi', 'hi ka re': 'hikare',
            'su ga ta': 'sugata', 'shizu ka': 'shizuka',
        }
        for incorrect, correct in pronunciation_fixes.items():
            text = text.replace(incorrect, correct)
        
        return text
    
    def add_proper_spacing(self, text: str) -> str:
        """Fix spacing and standardize Japanese particles."""
        # Adjective + noun compounds
        patterns = [
            (r'yasashi sa', r'yasashisa'), (r'([a-zāēīōū]+) sa\b', r'\1sa'),
            (r'azaya ka na', r'azayakana'), (r'aza ya ka na', r'azayakana'),
            (r'nosu igen', r'nosuigen'), (r'maru de', r'marude'),
            (r'wa kanai', r'hakanai'), (r'mu ne', r'mune'),
            (r'su ga ta', r'sugata'), (r'yo nan do', r'yonando'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        # Verb conjugations
        verb_patterns = [
            (r'([a-zāēīōū]+) (te|de) (i[a-zāēīōū]+)', r'\1\2\3'),
            (r'furue teru', r'furueteru'), (r'nomare te', r'nomarete'),
            (r'tsutsuma re ta', r'tsutsumareta'), (r'([a-zāēīōū]+) te\b', r'\1te'),
            (r'nokoshi te', r'nokoshite'), (r'sagashi te', r'sagashite'),
            (r'hi ka re', r'hikare'), (r'shizu ka ni', r'shizukani'),
        ]
        
        for pattern, replacement in verb_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Particle fixes
        text = re.sub(r'ga([a-zāēīōū])', r'ga \1', text)
        text = re.sub(r'watakushio', r'watakushi o', text)
        
        # Final cleanup
        text = re.sub(r' +', ' ', text).strip()
        text = text.replace(' ha ', ' wa ')
        if text.startswith('ha '): text = 'wa ' + text[3:]
        text = text.replace(' wo ', ' o ')
        text = text.replace(' he ', ' e ')
        
        return text
    
    def romanize(self, text: str, language: str = "ja") -> str:
        """
        Romanize Japanese text using fugashi + pykakasi.
        
        Args:
            text: Japanese text to romanize
            language: Source language (default: "ja")
            
        Returns:
            Romanized text
        """
        if language not in ("ja", "auto"):
            logger.warning(f"Local romanizer only supports Japanese, got: {language}")
            return text
        
        # Handle multi-line text (LRC format) by processing line by line
        if '\n' in text:
            lines = text.split('\n')
            romanized_lines = []
            for line in lines:
                if line.strip():
                    romanized_lines.append(self._romanize_single_line(line))
                else:
                    romanized_lines.append(line)  # Preserve empty lines
            result = '\n'.join(romanized_lines)
            # Clean LRC timestamps at the end
            return clean_lrc_timestamps(result)
        else:
            # Single line processing
            return self._romanize_single_line(text)
    
    def _romanize_single_line(self, text: str) -> str:
        """Romanize a single line of text."""
        nodes = self.tagger(text)
        romaji_parts = []
        
        for i, node in enumerate(nodes):
            pronunciation_kata = node.feature.kana or node.surface
            
            if not pronunciation_kata:
                continue
            
            # Convert katakana to romaji
            result = self.kks.convert(pronunciation_kata)
            romaji_part = "".join([item['hepburn'] for item in result])
            romaji_part = self.post_process_romaji(romaji_part)
            
            if romaji_part.strip():
                romaji_parts.append(romaji_part)
        
        # Join and process
        romaji_text = ' '.join(romaji_parts)
        romaji_text = re.sub(r' +', ' ', romaji_text).strip()
        romaji_text = romaji_text.replace(' ha ', ' wa ')
        if romaji_text.startswith('ha '): romaji_text = 'wa ' + romaji_text[3:]
        romaji_text = romaji_text.replace(' wo ', ' o ')
        romaji_text = romaji_text.replace('「', '"').replace('」', '"')
        romaji_text = self.add_proper_spacing(romaji_text)
        
        # Capitalize first letter
        if romaji_text and romaji_text[0].isalpha():
            romaji_text = romaji_text[0].upper() + romaji_text[1:]
        
        # Clean LRC timestamps (remove spaces inside and after timestamps)
        romaji_text = clean_lrc_timestamps(romaji_text)
        
        return romaji_text


class AIRomanizer(RomanizerBase):
    """AI-based romanization using OpenAI or Gemini APIs."""
    
    def __init__(
        self,
        provider: Literal["openai", "gemini"],
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI support requires: pip install openai")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self.model = model or "gpt-3.5-turbo"
        elif provider == "gemini":
            # We'll use direct REST API calls for Gemini (more reliable)
            # Use stable model instead of experimental
            self.model = model or "gemini-2.5-flash"
        
        logger.info(f"AI romanizer initialized with provider: {provider}, model: {self.model}")
    
    def romanize(self, text: str, language: str = "ja") -> str:
        """
        Romanize text using AI.
        
        Args:
            text: Text to romanize
            language: Source language
            
        Returns:
            Romanized text
        """
        # Check if this is LRC format (has timestamps)
        is_lrc = bool(re.search(r'\[\d+:\d+\.\d+\]', text))
        
        if is_lrc:
            prompt = f"""Romanize the following {language} lyrics in LRC format using accurate Hepburn romanization.

IMPORTANT:
- Preserve the exact timestamp format [mm:ss.xx]
- Keep each line on a separate line (preserve newlines)
- Only romanize the lyrics text, not the timestamps
- Use proper spacing between words
- Convert particles correctly (は→wa, を→o, へ→e)

Text:
{text}

Provide only the romanized text with timestamps, no explanations."""
        else:
            prompt = f"""Romanize the following {language} text using accurate Hepburn romanization.
Rules:
- Use proper spacing between words
- Convert particles correctly (は→wa, を→o, へ→e)
- Use macrons for long vowels where appropriate
- Keep natural word boundaries

Text: {text}

Provide only the romanized text, no explanations."""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                result_text = response.choices[0].message.content.strip()
                
                # Clean LRC timestamps (remove spaces inside and after timestamps)
                result_text = clean_lrc_timestamps(result_text)
                return result_text
            
            elif self.provider == "gemini":
                # Use REST API directly with retry logic
                import requests
                import json
                import urllib3
                import time
                
                # Suppress SSL warnings if needed
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
                headers = {
                    'Content-Type': 'application/json',
                }
                params = {
                    'key': self.api_key
                }
                data = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }]
                }
                
                # Retry logic with exponential backoff
                max_retries = 3
                base_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        response = requests.post(url, headers=headers, params=params, json=data, timeout=30, verify=False)
                        response.raise_for_status()
                        
                        result = response.json()
                        result_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                        
                        # Clean LRC timestamps (remove spaces inside and after timestamps)
                        result_text = clean_lrc_timestamps(result_text)
                        return result_text
                    
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 404:
                            logger.error(f"Gemini model '{self.model}' not found. Try 'gemini-1.5-flash' or 'gemini-pro'")
                            raise
                        elif e.response.status_code == 429:
                            if attempt < max_retries - 1:
                                delay = base_delay ** (attempt + 1)
                                logger.warning(f"Rate limited by Gemini API. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                                time.sleep(delay)
                            else:
                                logger.error("Max retries reached for Gemini API rate limiting")
                                raise
                        else:
                            raise
                    except Exception as e:
                        if attempt < max_retries - 1:
                            delay = base_delay ** (attempt + 1)
                            logger.warning(f"Gemini API error: {e}. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            raise
        
        except Exception as e:
            logger.error(f"AI romanization failed: {e}")
            raise


class Romanizer:
    """Main romanizer class with fallback support."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self.local_romanizer: Optional[LocalRomanizer] = None
        self.ai_romanizer: Optional[AIRomanizer] = None
        
        # Initialize based on config
        self._initialize_romanizers()
    
    def _initialize_romanizers(self):
        """Initialize romanizers based on configuration."""
        provider = self.config.api.default_provider
        
        # Always try to initialize local romanizer as fallback
        if LOCAL_ROMANIZATION_AVAILABLE:
            try:
                self.local_romanizer = LocalRomanizer()
            except Exception as e:
                logger.warning(f"Failed to initialize local romanizer: {e}")
        
        # Initialize AI romanizer if configured
        if provider == "openai" and self.config.api.openai_api_key:
            try:
                self.ai_romanizer = AIRomanizer(
                    provider="openai",
                    api_key=self.config.api.openai_api_key,
                    base_url=self.config.api.openai_base_url,
                    model=self.config.api.openai_model,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI romanizer: {e}")
        
        elif provider == "gemini" and self.config.api.gemini_api_key:
            try:
                self.ai_romanizer = AIRomanizer(
                    provider="gemini",
                    api_key=self.config.api.gemini_api_key,
                    base_url=self.config.api.gemini_base_url,
                    model=self.config.api.gemini_model,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini romanizer: {e}")
    
    def romanize(self, text: str, language: str = "ja", use_ai: bool = False) -> str:
        """
        Romanize text with automatic fallback.
        
        Args:
            text: Text to romanize
            language: Source language
            use_ai: Force AI romanization if available
            
        Returns:
            Romanized text
        """
        # Try AI first if requested and available
        if use_ai and self.ai_romanizer:
            try:
                return self.ai_romanizer.romanize(text, language)
            except Exception as e:
                logger.warning(f"AI romanization failed, falling back to local: {e}")
        
        # Fall back to local romanizer
        if self.local_romanizer:
            return self.local_romanizer.romanize(text, language)
        
        # If AI was not tried yet, try it now as last resort
        if not use_ai and self.ai_romanizer:
            try:
                return self.ai_romanizer.romanize(text, language)
            except Exception as e:
                logger.error(f"All romanization methods failed: {e}")
        
        raise RuntimeError("No romanization method available")
