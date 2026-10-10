# Editable Rules Directory

This directory contains human-editable configuration files that govern text normalization, phonetic transliteration, stopword filtering, and suffix stripping in `banglish-search`.

All files in this directory are loaded dynamically by the engine without hardcoding in core logic.

## Files

1. **`unicode_normalization.json`**
   - Maps visually identical or decomposed Unicode characters to canonical forms.
   - Handles Bengali Hasanta, Nukta, Dari, Bengali Khanda Ta (`ৎ`), Dari (`।`), and ligature variants.

2. **`phonetic_rules.json`**
   - Configures Banglish romanization to Bengali character/token generation rules.
   - Maps Latin digraphs/graphemes (e.g. `kh` -> `খ`, `th` -> `থ`, `bh` -> `ভ`, `sh` -> `শ`/`ষ`/`স`) and phonetic equivalence classes.

3. **`bangla_stopwords.txt`**
   - Newline-delimited list of common high-frequency Bengali grammatical stopwords to optionally penalize or filter during ranking.

4. **`bangla_suffixes.txt`**
   - Newline-delimited list of common Bengali inflectional and case markers (e.g. `গুলো`, `গুলি`, `দের`, `এর`, `কে`, `তে`, `রা`) for rule-based stemming/affix normalization.
