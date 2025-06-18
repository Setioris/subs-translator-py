import argparse
import logging
import os
import re
import sys
import torch
import gc
from pathlib import Path
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import srt
import pysubs2
import platform
import warnings
import psutil

# Filter out warnings
warnings.filterwarnings("ignore", message="Xet Storage is enabled for this repo")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Detect operating system
IS_WINDOWS = platform.system() == 'Windows'
IS_MAC = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'

# Set up UTF-8 encoding for Windows
if IS_WINDOWS:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Set up logging with UTF-8 support
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler('translation.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Stream handler with UTF-8 support
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logging()

# Supported languages mapping to NLLB codes
LANGUAGE_MAPPING = {
    'en': 'eng_Latn', 'eng': 'eng_Latn', 'english': 'eng_Latn',
    'ro': 'ron_Latn', 'rom': 'ron_Latn', 'romanian': 'ron_Latn',
    'ja': 'jpn_Jpan', 'jp': 'jpn_Jpan', 'japanese': 'jpn_Jpan',
    'fr': 'fra_Latn', 'fre': 'fra_Latn', 'french': 'fra_Latn',
    'de': 'deu_Latn', 'ger': 'deu_Latn', 'german': 'deu_Latn',
    'es': 'spa_Latn', 'spa': 'spa_Latn', 'spanish': 'spa_Latn',
    'it': 'ita_Latn', 'ita': 'ita_Latn', 'italian': 'ita_Latn',
    'ru': 'rus_Cyrl', 'rus': 'rus_Cyrl', 'russian': 'rus_Cyrl',
    'zh': 'zho_Hans', 'chi': 'zho_Hans', 'chinese': 'zho_Hans',
    'ko': 'kor_Hang', 'kor': 'kor_Hang', 'korean': 'kor_Hang',
    'ar': 'arb_Arab', 'ara': 'arb_Arab', 'arabic': 'arb_Arab',
    'hi': 'hin_Deva', 'hin': 'hin_Deva', 'hindi': 'hin_Deva',
    'pt': 'por_Latn', 'por': 'por_Latn', 'portuguese': 'por_Latn',
    'tr': 'tur_Latn', 'tur': 'tur_Latn', 'turkish': 'tur_Latn',
    'nl': 'nld_Latn', 'dut': 'nld_Latn', 'dutch': 'nld_Latn',
    'vi': 'vie_Latn', 'vie': 'vie_Latn', 'vietnamese': 'vie_Latn',
    'th': 'tha_Thai', 'tha': 'tha_Thai', 'thai': 'tha_Thai',
    'pl': 'pol_Latn', 'pol': 'pol_Latn', 'polish': 'pol_Latn',
    'uk': 'ukr_Cyrl', 'ukr': 'ukr_Cyrl', 'ukrainian': 'ukr_Cyrl',
    'el': 'ell_Grek', 'gre': 'ell_Grek', 'greek': 'ell_Grek',
    'he': 'heb_Hebr', 'heb': 'heb_Hebr', 'hebrew': 'heb_Hebr',
}

# Language-specific generation parameters
LANG_GENERATION_PARAMS = {
    # Japanese needs longer sequences and less repetition restriction
    'jpn_Jpan': {
        'max_length': 600,
        'no_repeat_ngram_size': 2,
        'num_beams': 5,
        'early_stopping': False
    },
    # Romanian benefits from more beams and longer sequences
    'ron_Latn': {
        'max_length': 550,
        'num_beams': 5,
        'no_repeat_ngram_size': 3,
        'early_stopping': True
    },
    # English uses standard parameters
    'eng_Latn': {
        'max_length': 512,
        'num_beams': 4,
        'no_repeat_ngram_size': 3,
        'early_stopping': True
    },
    # Default parameters for other languages
    'default': {
        'max_length': 512,
        'num_beams': 4,
        'no_repeat_ngram_size': 3,
        'early_stopping': True
    }
}

class SubtitleTranslator:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        model_size: str = "600M",
        device: str = "auto",
        threads: int = 0,
        quality_mode: str = "balanced"  # New parameter for quality control
    ):
        """
        Enhanced multi-language subtitle translator
        :param source_lang: Source language code (e.g., 'en', 'ja')
        :param target_lang: Target language code (e.g., 'ro', 'fr')
        :param model_size: NLLB model size - "418M", "600M", "1.3B", or "3.3B"
        :param device: "auto", "cpu", or "cuda"
        :param threads: Number of CPU threads to use (0 = automatic)
        :param quality_mode: Translation quality mode - "fast", "balanced", or "high"
        """
        # Configure device with robust fallback
        self.device = self._determine_device(device)

        # Configure CPU threading
        self.threads = threads
        if threads > 0 and self.device == "cpu":
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["MKL_NUM_THREADS"] = str(threads)
            if not IS_WINDOWS:  # Windows has issues with torch.set_num_threads
                torch.set_num_threads(threads)
            logger.info(f"CPU optimization: Using {threads} threads")

        # Map languages to NLLB codes
        self.source_lang = self._map_language(source_lang)
        self.target_lang = self._map_language(target_lang)

        # Model selection
        model_choices = {
            "418M": "facebook/nllb-200-distilled-418M",
            "600M": "facebook/nllb-200-distilled-600M",
            "1.3B": "facebook/nllb-200-1.3B",
            "3.3B": "facebook/nllb-200-3.3B"
        }
        self.model_name = model_choices.get(model_size)
        if not self.model_name:
            raise ValueError(f"Invalid model size: {model_size}. Choose '418M', '600M', '1.3B' or '3.3B'")

        # Quality mode configuration
        self.quality_mode = quality_mode
        if quality_mode == "fast":
            self.quality_factor = 0.7
        elif quality_mode == "balanced":
            self.quality_factor = 1.0
        elif quality_mode == "high":
            self.quality_factor = 1.5
        else:
            raise ValueError(f"Invalid quality mode: {quality_mode}. Choose 'fast', 'balanced', or 'high'")

        # Use ASCII arrow for Windows compatibility
        arrow = "->" if IS_WINDOWS else "→"
        logger.info(
            f"Initializing translator | Device: {self.device.upper()} | "
            f"Model: {self.model_name} | {self.source_lang} {arrow} {self.target_lang} | "
            f"Quality: {quality_mode}"
        )

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Disable tokenizer parallelism to prevent deadlocks on Windows
            if IS_WINDOWS:
                self.tokenizer.model_max_length = 512
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Get forced token ID for target language (must come before model warm-up)
            self.forced_bos_token_id = self._get_forced_token_id()

            # Load model with robust device handling
            self.model = self._load_model(model_size)

            # Warm up model explicitly after everything is ready
            self._warm_up_model(self.model)

            logger.info("Model loaded successfully")

            # Initialize memory tracking
            self.max_memory_usage = 0
            self.initial_memory = self._get_memory_usage()

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 3)
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 3)

    def _log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        current = self._get_memory_usage()
        self.max_memory_usage = max(self.max_memory_usage, current)
        usage = current - self.initial_memory
        logger.info(f"Memory usage {context}: {usage:.2f} GB (Peak: {self.max_memory_usage:.2f} GB)")

    def _get_forced_token_id(self):
        """Get the forced token ID using the correct method for NLLB tokenizers"""
        # Format the language token correctly for NLLB
        lang_token = f"{self.target_lang}"
        try:
            # Try to get the token ID directly
            if hasattr(self.tokenizer, "lang_code_to_id"):
                return self.tokenizer.lang_code_to_id[lang_token]
            # Fallback method if attribute doesn't exist
            return self.tokenizer.convert_tokens_to_ids(lang_token)
        except Exception as e:
            logger.error(f"Error getting token ID for {lang_token}: {e}")
            # Fallback to English if target fails
            if self.target_lang != "eng_Latn":
                logger.warning(f"Falling back to English token for {lang_token}")
                return self.tokenizer.lang_code_to_id.get("eng_Latn", self.tokenizer.convert_tokens_to_ids("eng_Latn"))
            raise ValueError(f"Could not get token ID for language: {lang_token}") from e

    def _determine_device(self, device: str) -> str:
        """Enhanced device detection with Windows-specific checks"""
        if device == "auto":
            if torch.cuda.is_available():
                # Windows-specific memory check
                if IS_WINDOWS:
                    try:
                        # Check for VRAM capacity
                        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                        if total_vram < 4:
                            logger.info(f"CUDA available but only {total_vram:.1f}GB VRAM. Using CPU instead.")
                            return "cpu"
                    except Exception as e:
                        logger.warning(f"CUDA detection failed: {e}. Falling back to CPU.")
                        return "cpu"
                return "cuda"
            return "cpu"
        elif device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device

    def _load_model(self, model_size: str):
        """Load model with proper configuration and fallback"""
        try:
            # Determine precision based on device
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Load model with error handling
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype
            )

            # Move to device with fallback
            try:
                model = model.to(self.device)
                return model
            except RuntimeError as e:
                if "CUDA" in str(e) or "memory" in str(e):
                    logger.warning(f"CUDA error: {e}. Trying CPU fallback.")
                    self.device = "cpu"
                    return model.to("cpu")
                raise
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Could not load model: {e}")

    def _warm_up_model(self, model):
        """Warm up the model with a small translation"""
        try:
            test_text = "Model warm-up"
            inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                truncation=True,
                max_length=32
            ).to(self.device)
            
            model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=32,
                num_beams=1
            )
            torch.cuda.empty_cache() if self.device == "cuda" else None
            gc.collect()
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _map_language(self, lang: str) -> str:
        """Map common language codes to NLLB language codes"""
        lang = lang.lower()
        if lang in LANGUAGE_MAPPING:
            return LANGUAGE_MAPPING[lang]
        
        # Try to find matching language code
        for code, nllb_code in LANGUAGE_MAPPING.items():
            if lang in code or lang in nllb_code:
                return nllb_code
        
        # Assume it's already an NLLB code
        if "_" in lang and len(lang.split("_")) == 2:
            return lang
        
        raise ValueError(f"Unsupported language: {lang}. Please use 2-letter codes or NLLB format")

    def _get_batch_size(self, model_size: str) -> int:
        """Determine batch size based on device, model size, and quality mode"""
        base_batch_size = {
            "418M": {"cuda": 24, "cpu": 12},
            "600M": {"cuda": 16, "cpu": 8},
            "1.3B": {"cuda": 8, "cpu": 4},
            "3.3B": {"cuda": 4, "cpu": 2}
        }.get(model_size, {"cuda": 8, "cpu": 4})
        
        # Adjust based on quality mode
        quality_factor = {
            "fast": 1.2,
            "balanced": 1.0,
            "high": 0.8
        }.get(self.quality_mode, 1.0)
        
        # Get base size for current device
        device_type = "cuda" if self.device == "cuda" else "cpu"
        size = max(1, int(base_batch_size[device_type] * quality_factor))
        
        # Further reduce for Windows
        if IS_WINDOWS and device_type == "cuda":
            size = max(1, size // 2)
            
        return size

    def _get_generation_params(self) -> Dict:
        """Get generation parameters based on target language and quality mode"""
        # Get language-specific parameters or use defaults
        params = LANG_GENERATION_PARAMS.get(self.target_lang, LANG_GENERATION_PARAMS['default']).copy()
        
        # Adjust based on quality mode
        if self.quality_mode == "fast":
            params['num_beams'] = max(2, int(params['num_beams'] * 0.7))
            params['no_repeat_ngram_size'] = max(2, int(params['no_repeat_ngram_size'] * 0.8))
        elif self.quality_mode == "high":
            params['num_beams'] = int(params['num_beams'] * 1.3)
            params['max_length'] = int(params['max_length'] * 1.2)
        
        # Add forced token
        params['forced_bos_token_id'] = self.forced_bos_token_id
        
        return params

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Efficient batch translation with memory management"""
        if not texts:
            return []
            
        try:
            # Set tokenizer source language
            self.tokenizer.src_lang = self.source_lang
            
            # Tokenize inputs
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get generation parameters
            gen_params = self._get_generation_params()
            
            # Generate translations
            translated_tokens = self.model.generate(
                **inputs,
                **gen_params
            )
            
            # Decode results
            results = self.tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True
            )
            
            # Memory cleanup
            del inputs
            del translated_tokens
            torch.cuda.empty_cache() if self.device == "cuda" else None
            gc.collect()
            
            return results
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "memory" in str(e):
                # Reduce batch size on memory error
                new_batch_size = max(1, self.batch_size // 2)
                logger.warning(f"Memory error, reducing batch size from {self.batch_size} to {new_batch_size}")
                self.batch_size = new_batch_size
                
                # Clear memory
                torch.cuda.empty_cache() if self.device == "cuda" else None
                gc.collect()
                
                return self.translate_batch(texts)
            else:
                logger.error(f"Batch translation failed: {e}")
                # Fallback to individual translation
                return [self.translate_text(text) for text in texts]
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Fallback to individual translation
            return [self.translate_text(text) for text in texts]

    def translate_text(self, text: str) -> str:
        """Translate individual text with fallback"""
        if not text.strip():
            return text
            
        try:
            # Set tokenizer source language
            self.tokenizer.src_lang = self.source_lang
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get generation parameters
            gen_params = self._get_generation_params()
            
            # Generate translation
            translated_tokens = self.model.generate(
                **inputs,
                **gen_params
            )
            
            return self.tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )
            
        except Exception as e:
            logger.error(f"Failed to translate: '{text[:50]}...' - {e}")
            return text  # Return original as fallback

    def process_srt(self, input_path: str, output_path: str):
        """Process SRT files with enhanced memory management"""
        logger.info(f"Processing SRT file: {input_path}")
        
        # Detect file encoding
        encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'utf-16']
        content = None
        
        for encoding in encodings:
            try:
                with open(input_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"Detected encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise UnicodeDecodeError(f"Failed to decode file: {input_path}")
        
        # Parse subtitles
        subs = list(srt.parse(content))
        logger.info(f"Found {len(subs)} subtitles")
        
        # Extract texts for translation
        original_texts = [sub.content for sub in subs]
        
        # Determine batch size
        model_size = self.model_name.split('-')[-1]
        self.batch_size = self._get_batch_size(model_size)
        logger.info(f"Using batch size: {self.batch_size} for model size: {model_size}")
        
        # Translate in batches with progress
        translated_texts = []
        total_batches = (len(original_texts) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=total_batches, desc="Translating", unit="batch") as pbar:
            for i in range(0, len(original_texts), self.batch_size):
                batch = original_texts[i:i + self.batch_size]
                translated_batch = self.translate_batch(batch)
                translated_texts.extend(translated_batch)
                pbar.update(1)
                self._log_memory_usage(f"after batch {i//self.batch_size + 1}/{total_batches}")
        
        # Create translated subtitles
        translated_subs = []
        for i, sub in enumerate(subs):
            # Preserve original line breaks
            translated_content = translated_texts[i]
            if '\n' in sub.content:
                # Try to preserve original line break structure
                original_lines = sub.content.split('\n')
                translated_lines = translated_content.split('\n')
                
                # Use original line breaks if counts match
                if len(original_lines) == len(translated_lines):
                    translated_content = '\n'.join(translated_lines)
                else:
                    # Fallback to single line if structure differs
                    translated_content = ' '.join(translated_lines)
            
            translated_subs.append(srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=translated_content,
                proprietary=sub.proprietary
            ))
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(translated_subs))
        
        logger.info(f"Translated SRT saved: {output_path}")

    def process_ass(self, input_path: str, output_path: str):
        """Process ASS files with formatting preservation"""
        logger.info(f"Processing ASS file: {input_path}")
        
        # Load subtitles with encoding detection
        try:
            subs = pysubs2.load(input_path)
        except UnicodeDecodeError:
            # Try common encodings if default fails
            encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    subs = pysubs2.load(input_path, encoding=encoding)
                    logger.info(f"Detected encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError(f"Failed to decode file: {input_path}")
        
        logger.info(f"Found {len(subs)} events")
        
        # Extract dialogue texts
        original_texts = []
        event_indices = []
        formatting_tags = []
        
        for i, event in enumerate(subs.events):
            if event.type == "Dialogue":
                # Extract and store formatting tags
                tags = re.findall(r'\{.*?\}', event.text)
                formatting_tags.append(tags)
                
                # Clean ASS formatting tags
                clean_text = re.sub(r'\{.*?\}', '', event.text)
                original_texts.append(clean_text)
                event_indices.append(i)
        
        # Determine batch size
        model_size = self.model_name.split('-')[-1]
        self.batch_size = self._get_batch_size(model_size)
        logger.info(f"Using batch size: {self.batch_size} for model size: {model_size}")
        
        # Translate dialogue texts
        translated_texts = []
        total_batches = (len(original_texts) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=total_batches, desc="Translating", unit="batch") as pbar:
            for i in range(0, len(original_texts), self.batch_size):
                batch = original_texts[i:i + self.batch_size]
                translated_batch = self.translate_batch(batch)
                translated_texts.extend(translated_batch)
                pbar.update(1)
                self._log_memory_usage(f"after batch {i//self.batch_size + 1}/{total_batches}")
        
        # Apply translations while preserving original formatting
        for idx, text, tags in zip(event_indices, translated_texts, formatting_tags):
            # Preserve original formatting tags
            subs.events[idx].text = ''.join(tags) + text if tags else text
        
        # Save output
        subs.save(output_path, encoding='utf-8')
        logger.info(f"Translated ASS saved: {output_path}")

    def translate_subtitle_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """Main translation workflow with enhanced validation"""
        try:
            # Validate input file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            # Determine output path
            if output_path is None:
                input_file = Path(input_path)
                lang_suffix = self.target_lang.split('_')[0]  # Use only language part
                output_path = str(input_file.with_name(
                    f"{input_file.stem}_{lang_suffix}{input_file.suffix}"
                ))
                
            # Process based on file type
            if input_path.lower().endswith('.srt'):
                self.process_srt(input_path, output_path)
            elif input_path.lower().endswith(('.ass', '.ssa')):
                self.process_ass(input_path, output_path)
            else:
                raise ValueError("Unsupported file format. Only .srt, .ass, and .ssa are supported.")
                
            return output_path
            
        except Exception as e:
            logger.exception("Translation failed")
            raise RuntimeError(f"Translation failed: {str(e)}")

def print_supported_languages():
    """Print supported languages in a readable format"""
    print("\nSUPPORTED LANGUAGES:")
    unique_languages = {}
    for key, value in LANGUAGE_MAPPING.items():
        if value not in unique_languages:
            unique_languages[value] = []
        unique_languages[value].append(key)
    
    for nllb_code, codes in unique_languages.items():
        main_code = codes[0]
        aliases = ", ".join(codes[1:])
        print(f"  {main_code.upper().ljust(8)} → {nllb_code.ljust(12)} (aliases: {aliases})")

def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Professional Multi-Language Subtitle Translator using Facebook's NLLB models",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""EXAMPLES:
  # Basic English to Romanian translation
  python subtitle_translator.py movie.srt --source en --target ro
  
  # Japanese to Romanian with 3.3B model and high quality
  python subtitle_translator.py anime.ass --source ja --target ro --model-size 3.3B --quality high
  
  # French to English using GPU
  python subtitle_translator.py french_movie.srt --source fr --target en --device cuda --threads 4
  
  # Custom output path
  python subtitle_translator.py input.srt --source de --target it --output german_to_italian.srt
  
  # Show supported languages
  python subtitle_translator.py --list-languages

TIPS:
  • First run will download models (~1.2GB for 600M, ~6.5GB for 3.3B)
  • Use --device cuda for GPU acceleration (requires compatible NVIDIA GPU)
  • For large files, use the 1.3B or 3.3B model for better quality
  • On Windows, use smaller batch sizes to prevent freezing
  • For Japanese → Romanian, use high quality mode for best results
"""
    )
    parser.add_argument('input_path', nargs='?', help='Path to subtitle file (SRT, ASS, SSA)')
    parser.add_argument('--source', required=True, help='Source language code (e.g., en, ja)')
    parser.add_argument('--target', required=True, help='Target language code (e.g., ro, fr)')
    parser.add_argument('--output', help='Custom output file path')
    parser.add_argument('--model-size', choices=['418M', '600M', '1.3B', '3.3B'], default='600M',
                        help='Model size to use (418M=fastest, 3.3B=best quality)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Processing device (auto=detect best)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of CPU threads (0=auto, for CPU mode only)')
    parser.add_argument('--quality', choices=['fast', 'balanced', 'high'], default='balanced',
                        help='Translation quality mode (fast/balanced/high)')
    parser.add_argument('--list-languages', action='store_true',
                        help='Show all supported languages and exit')
    
    args = parser.parse_args()
    
    # Handle language list request
    if args.list_languages:
        print_supported_languages()
        print("\nNote: You can use any 2-letter code or full language name shown above")
        return 0
        
    # Validate required arguments for translation
    if not args.input_path:
        parser.error("Missing INPUT_PATH argument\nUse --help for usage instructions")
    
    try:
        # Use ASCII arrow for Windows compatibility
        arrow = "->" if IS_WINDOWS else "→"
        logger.info(f"Starting translation: {args.input_path} [{args.source} {arrow} {args.target}]")
        logger.info(f"Platform: {platform.system()} {platform.release()}")
        logger.info(f"Python: {sys.version.split()[0]}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"Device: {args.device} | Model: {args.model_size} | Quality: {args.quality}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        if args.device == "cuda" and not cuda_available:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        
        translator = SubtitleTranslator(
            source_lang=args.source,
            target_lang=args.target,
            model_size=args.model_size,
            device=args.device,
            threads=args.threads,
            quality_mode=args.quality
        )
        output_path = translator.translate_subtitle_file(
            input_path=args.input_path,
            output_path=args.output
        )
        print(f"\n✅ Translation saved to: {output_path}")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"❌ Critical failure: {str(e)}")
        
        # Provide troubleshooting tips for tokenizer issues
        if "lang_code_to_id" in str(e) or "attribute" in str(e):
            print("\nTokenizer Troubleshooting Tips:")
            print("1. Update transformers library: pip install --upgrade transformers")
            print("2. Try using a different model size: --model-size 600M")
            print("3. Check for known issues: https://github.com/huggingface/transformers/issues")
        
        # Provide troubleshooting tips for CUDA issues
        elif "CUDA" in str(e) or "cuda" in str(e):
            print("\nCUDA Troubleshooting Tips:")
            print("1. Verify you have an NVIDIA GPU with CUDA support")
            print("2. Install CUDA-compatible PyTorch: https://pytorch.org/get-started/locally/")
            print("3. Use --device cpu to force CPU-only mode")
            print("4. Check driver version with: nvidia-smi")
            print("5. Try smaller model: --model-size 418M")
        
        # Windows-specific troubleshooting tips
        elif IS_WINDOWS:
            print("\nWindows Troubleshooting Tips:")
            print("1. Try reducing batch size by using --quality fast")
            print("2. Use --device cpu if you have GPU memory issues")
            print("3. Update your graphics drivers")
            print("4. Use the smaller model: --model-size 418M")
        
        return 1

if __name__ == "__main__":
    # Windows-specific configuration
    if IS_WINDOWS:
        # Improve Windows compatibility
        os.environ["PYTHONIOENCODING"] = "utf-8"
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    
    # Clear memory before starting
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    main()
