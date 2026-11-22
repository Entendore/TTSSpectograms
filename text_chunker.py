# text_chunker.py
"""
Text chunking utility for TTS backends.
Splits long text into sentence-based chunks that fit model limits.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    max_chars: int = 500
    max_sentences: int = 5
    min_chunk_chars: int = 50
    language: str = "English"


_SPEAKING_RATES = {
    "English": 15.0, "Chinese": 6.0, "Japanese": 8.0, "Korean": 8.0,
    "German": 13.0, "French": 14.0, "Russian": 12.0, "Portuguese": 13.0,
    "Spanish": 14.0, "Italian": 14.0,
}

_SENTENCE_PATTERNS = {
    "Chinese": r'[。！？；\n]',
    "Japanese": r'[。！？\n]',
    "Korean": r'[。！？\n]',
    "default": r'(?<=[.!?;])\s+|\n+',
}


def get_chunker_for_language(language: str, config: ChunkingConfig = None) -> "TextChunker":
    if config is None:
        config = ChunkingConfig(language=language)
    else:
        config.language = language
    return TextChunker(config)


class TextChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config

    def needs_chunking(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        return len(text.strip()) > self.config.max_chars

    def chunk_text(self, text: str, language: str = None) -> List[Tuple[str, float]]:
        lang = language or self.config.language
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text, lang)
        if not sentences:
            return [(text, self._estimate_duration(text, lang))]

        chunks: List[Tuple[str, float]] = []
        current_sentences: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue

            if len(sent_stripped) > self.config.max_chars:
                if current_sentences:
                    chunk_text = self._join_sentences(current_sentences, lang)
                    chunks.append((chunk_text, self._estimate_duration(chunk_text, lang)))
                    current_sentences = []
                    current_len = 0

                sub_chunks = self._split_long_sentence(sent_stripped, lang)
                for sc in sub_chunks:
                    chunks.append((sc, self._estimate_duration(sc, lang)))
                continue

            if (current_len + len(sent_stripped) + 1 > self.config.max_chars
                    or len(current_sentences) >= self.config.max_sentences):
                if current_sentences:
                    chunk_text = self._join_sentences(current_sentences, lang)
                    chunks.append((chunk_text, self._estimate_duration(chunk_text, lang)))
                    current_sentences = []
                    current_len = 0

            current_sentences.append(sent_stripped)
            current_len += len(sent_stripped) + 1

        if current_sentences:
            chunk_text = self._join_sentences(current_sentences, lang)
            chunks.append((chunk_text, self._estimate_duration(chunk_text, lang)))

        if len(chunks) >= 2:
            last_text, last_dur = chunks[-1]
            if len(last_text) < self.config.min_chunk_chars:
                prev_text, prev_dur = chunks[-2]
                merged = self._join_sentences([prev_text, last_text], lang)
                chunks[-2] = (merged, self._estimate_duration(merged, lang))
                chunks.pop()

        return chunks if chunks else [(text, self._estimate_duration(text, lang))]

    def _split_sentences(self, text: str, language: str) -> List[str]:
        pattern = _SENTENCE_PATTERNS.get(language, _SENTENCE_PATTERNS["default"])
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def _split_long_sentence(self, sentence: str, language: str) -> List[str]:
        if language in ("Chinese", "Japanese", "Korean"):
            clause_pattern = r'[，、,；]'
        else:
            clause_pattern = r'[,;:]'

        clauses = re.split(clause_pattern, sentence)
        clauses = [c.strip() for c in clauses if c.strip()]

        if not clauses:
            return [sentence[i:i + self.config.max_chars]
                    for i in range(0, len(sentence), self.config.max_chars)]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for clause in clauses:
            if current_len + len(clause) + 1 > self.config.max_chars and current:
                chunks.append(self._join_sentences(current, lang=language))
                current = []
                current_len = 0
            current.append(clause)
            current_len += len(clause) + 1

        if current:
            chunks.append(self._join_sentences(current, lang=language))

        return chunks if chunks else [sentence]

    def _join_sentences(self, sentences: List[str], lang: str = "English") -> str:
        if lang in ("Chinese", "Japanese", "Korean"):
            return "".join(sentences)
        return " ".join(sentences)

    def _estimate_duration(self, text: str, language: str) -> float:
        rate = _SPEAKING_RATES.get(language, _SPEAKING_RATES["English"])
        return len(text) / rate