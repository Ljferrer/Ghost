import re
import json
import spacy

from pathlib import Path
from typing import List, Dict, Union
from spacy_langdetect import LanguageDetector
from pytorch_pretrained_bert import BertTokenizer

__all__ = ['LyricGeniusFormatter']


class LyricGeniusFormatter:
    Section = List[List[str]]
    Cleaned = Dict[str, Union[str, Section]]

    def __init__(self, tokenizer_type: str = 'bert-base-uncased',
                 do_lower_case: bool = True):
        self.nlp = spacy.load('en')
        self.nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        self.toke = BertTokenizer.from_pretrained(tokenizer_type,
                                                  do_lower_case=do_lower_case)

        # For splitting by \n\n followed by
        # [... , (...) , {... , int... , ...: , or (R/r)epeat...
        self.header_seed = '(\n\n(\[.*|\(.*\)|\{.*|[0-9].*|.*[:|: ]\n|.*(R|r)epeat.*))'
        # For cleaning up any missed characters
        self.clean_seed = '\([^)].*\)|\[.*?\]|\(|\)|\[|\]|:'

    def detect_lang(self, text: str):
        doc = self.nlp(text)
        return doc._.language

    def clean_sections(self, raw_lyrics: str) -> Section:
        sections = list()

        raw_sections = re.split(self.header_seed, '\n\n' + raw_lyrics)   # Catch [Intro]
        for raw_sect in raw_sections:
            clean_sect = re.sub(self.clean_seed, '', str(raw_sect))      # Clean residual
            split_sect = [l for l in clean_sect.split('\n') if len(self.toke.tokenize(l))]
            if len(split_sect) > 1:     # Only includes couplets or longer
                sections.append(split_sect)

        return sections

    # noinspection PyTypeChecker
    def format_lyrics(self, raw_lyrics_file: Path) -> Cleaned:

        # Load song dict
        with open(raw_lyrics_file, 'r') as rlf:
            raw = json.loads(rlf.read())
        raw_lyrics = str(raw['songs'][0]['lyrics'])

        # Detect language
        detect = self.detect_lang(raw_lyrics)
        if 'en' not in detect['language'] or detect['score'] < 0.80:
            return

        # Copy fields
        formatted = dict()
        formatted['artist'] = raw['artist']
        formatted['title'] = raw['songs'][0]['title']
        formatted['year'] = raw['songs'][0]['year']
        formatted['image'] = raw['songs'][0]['image']
        formatted['raw_lyrics'] = raw_lyrics

        # Clean lyrics
        formatted['sections'] = self.clean_sections(raw_lyrics)

        return formatted
