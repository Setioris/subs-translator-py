import argparse
import logging
import os
import re
import torch
from pathlib import Path
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import srt
import pysubs2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
}

class SubtitleTranslator:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        model_size: str = "600M",
        device: str = "auto",
        threads: int = 0
    ):
        """
        Multi-language subtitle translator using NLLB models
        :param source_lang: Source language code (e.g., 'en', 'ja')
        :param target_lang: Target language code (e.g., 'ro', 'fr')
        :param model_size: NLLB model size - "600M" or "3.3B"
        :param device: "auto", "cpu", or "cuda"
        :param threads: Number of CPU threads to use (0 = automatic)
        """
        # Configure device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Configure CPU threading
        if threads > 0 and self.device == "cpu":
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["MKL_NUM_THREADS"] = str(threads)
            torch.set_num_threads(threads)
            logger.info(f"CPU optimization: Using {threads} threads")
        
        # Map languages to NLLB codes
        self.source_lang = self._map_language(source_lang)
        self.target_lang = self._map_language(target_lang)
        
        # Model selection
        model_choices = {
            "600M": "facebook/nllb-200-distilled-600M",
            "3.3B": "facebook/nllb-200-3.3B"
        }
        self.model_name = model_choices.get(model_size)
        if not self.model_name:
            raise ValueError(f"Invalid model size: {model_size}. Choose '600M' or '3.3B'")
        
        # Batch size configuration
        if self.device == "cuda":
            self.batch_size = 16 if model_size == "600M" else 8
        else:
            self.batch_size = 8 if model_size == "600M" else 4

        logger.info(
            f"Initializing translator | Device: {self.device.upper()} | "
            f"Model: {self.model_name} | {self.source_lang} → {self.target_lang}"
        )
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine precision based on device
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype
            ).to(self.device)
            
            # Get forced token ID for target language
            self.forced_bos_token_id = self.tokenizer.lang_code_to_id[self.target_lang]
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

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

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Efficient batch translation"""
        if not texts:
            return []
            
        try:
            # Format inputs with source language prefix
            formatted_texts = [f"{self.source_lang} {text}" for text in texts]
            
            # Tokenize inputs
            inputs = self.tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translations
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=512,
                num_beams=3
            )
            
            # Decode results
            return self.tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True
            )
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Fallback to individual translation
            return [self.translate_text(text) for text in texts]

    def translate_text(self, text: str) -> str:
        """Translate individual text with fallback"""
        if not text.strip():
            return text
            
        try:
            # Format input with source language prefix
            formatted_text = f"{self.source_lang} {text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=512,
                num_beams=3
            )
            
            return self.tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )
            
        except Exception as e:
            logger.error(f"Failed to translate: '{text[:50]}...' - {e}")
            return text  # Return original as fallback

    def process_srt(self, input_path: str, output_path: str):
        """Process SRT files using professional srt library"""
        logger.info(f"Processing SRT file: {input_path}")
        
        # Read input file with encoding detection
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # Parse subtitles
        subs = list(srt.parse(content))
        logger.info(f"Found {len(subs)} subtitles")
        
        # Extract texts for translation
        original_texts = [sub.content for sub in subs]
        
        # Translate in batches with progress
        translated_texts = []
        
        for i in tqdm(range(0, len(original_texts), self.batch_size), 
                      desc="Translating", unit="batch"):
            batch = original_texts[i:i+self.batch_size]
            translated_batch = self.translate_batch(batch)
            translated_texts.extend(translated_batch)
        
        # Create translated subtitles
        translated_subs = []
        for i, sub in enumerate(subs):
            translated_subs.append(srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=translated_texts[i],
                proprietary=sub.proprietary
            ))
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(translated_subs))
        
        logger.info(f"Translated SRT saved: {output_path}")

    def process_ass(self, input_path: str, output_path: str):
        """Process ASS files using professional pysubs2 library"""
        logger.info(f"Processing ASS file: {input_path}")
        
        # Load subtitles
        subs = pysubs2.load(input_path)
        logger.info(f"Found {len(subs)} events")
        
        # Extract dialogue texts
        original_texts = []
        event_indices = []
        
        for i, event in enumerate(subs.events):
            if event.type == "Dialogue":
                # Clean ASS formatting tags
                clean_text = re.sub(r'\{.*?\}', '', event.text)
                original_texts.append(clean_text)
                event_indices.append(i)
        
        # Translate dialogue texts
        translated_texts = []
        
        for i in tqdm(range(0, len(original_texts), self.batch_size), 
                      desc="Translating", unit="batch"):
            batch = original_texts[i:i+self.batch_size]
            translated_batch = self.translate_batch(batch)
            translated_texts.extend(translated_batch)
        
        # Apply translations while preserving original formatting
        for idx, text in zip(event_indices, translated_texts):
            # Preserve original formatting tags
            original_text = subs.events[idx].text
            tags = re.findall(r'\{.*?\}', original_text)
            subs.events[idx].text = ''.join(tags) + text if tags else text
        
        # Save output
        subs.save(output_path, encoding='utf-8')
        logger.info(f"Translated ASS saved: {output_path}")

    def translate_subtitle_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """Main translation workflow"""
        try:
            # Validate input file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            # Determine output path
            if output_path is None:
                input_file = Path(input_path)
                output_path = str(input_file.with_name(
                    f"{input_file.stem}_{self.target_lang}{input_file.suffix}"
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
        description="🌍 Professional Multi-Language Subtitle Translator using Facebook's NLLB models",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""EXAMPLES:
  # Basic English to Romanian translation
  python subtitle_translator.py movie.srt --source en --target ro
  
  # Japanese to Romanian with 3.3B model
  python subtitle_translator.py anime.ass --source ja --target ro --model-size 3.3B
  
  # French to English using GPU
  python subtitle_translator.py french_movie.srt --source fr --target en --device cuda
  
  # Custom output path
  python subtitle_translator.py input.srt --source de --target it --output german_to_italian.srt
  
  # Show supported languages
  python subtitle_translator.py --list-languages

TIPS:
  • First run will download models (~1.2GB for 600M, ~6.5GB for 3.3B)
  • Use --device cuda for GPU acceleration (requires compatible NVIDIA GPU)
  • For large files, use the 3.3B model for better quality (slower)
  • ASS/SSA files preserve original formatting tags during translation
"""
    )
    parser.add_argument('input_path', nargs='?', help='Path to subtitle file (SRT, ASS, SSA)')
    parser.add_argument('--source', help='Source language code (e.g., en, ja)')
    parser.add_argument('--target', help='Target language code (e.g., ro, fr)')
    parser.add_argument('--output', help='Custom output file path')
    parser.add_argument('--model-size', choices=['600M', '3.3B'], default='600M',
                        help='Model size to use (600M=faster, 3.3B=better quality)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Processing device (auto=detect best)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of CPU threads (0=auto, for CPU mode only)')
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
        
    if not args.source or not args.target:
        parser.error("Both --source and --target arguments are required\nUse --help for usage instructions")
    
    try:
        logger.info(f"Starting translation: {args.input_path} [{args.source} → {args.target}]")
        translator = SubtitleTranslator(
            source_lang=args.source,
            target_lang=args.target,
            model_size=args.model_size,
            device=args.device,
            threads=args.threads
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
        return 1

if __name__ == "__main__":
    main()