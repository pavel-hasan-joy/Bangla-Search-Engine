"""Comprehensive pytest test suite for bsearch.unicode module.

Covers:
- Conjunct grapheme segmentation & tokenization (ক্ষ, স্ক্র, র্ক, ক্য, ত্ব, etc.)
- য় / ড় / ঢ় variants and canonical forms
- Split ো / ৌ vowel composition
- ZWNJ / ZWJ normalization policy
- Bangla digits to ASCII
- Mixed Bangla-English handling
- Mis-ordered pre-base vowel signs (িশক্ষা -> শিক্ষা, েখলা -> খেলা, etc.)
- Repeated hasanta collapsing
- Modifiers (ং, ঃ, ঁ) attachment
- Hyphen and apostrophe handling
- Suffix-driven stemming behind flag and minimum stem length protection
"""

import pytest

from bsearch.unicode import (
    ZWJ,
    ZWNJ,
    canonicalize_nukta,
    compose_two_part_vowels,
    convert_digits_to_ascii,
    normalize,
    repair_misordered_vowels,
    segment_graphemes,
    stem_word,
    tokenize,
)


# =========================================================================
# 1. Grapheme Segmenter & Conjuncts: ক্ষ, স্ক্র, র্ক, ক্য, ত্ব, etc.
# =========================================================================

def test_grapheme_khiyo_conjunct():
    """ক্ষ (ক + ্ + ষ) must stay intact as a single grapheme."""
    graphemes = segment_graphemes("ক্ষ")
    assert graphemes == ["ক্ষ"]


def test_grapheme_skro_triple_conjunct():
    """স্ক্র (স + ্ + ক + ্ + র) must stay intact as a single grapheme."""
    graphemes = segment_graphemes("স্ক্র")
    assert graphemes == ["স্ক্র"]


def test_grapheme_ro_reph_conjunct():
    """র্ক (র + ্ + ক) must stay intact as a single grapheme."""
    graphemes = segment_graphemes("র্ক")
    assert graphemes == ["র্ক"]


def test_grapheme_ya_phala_conjunct():
    """ক্য (ক + ্ + য) must stay intact as a single grapheme."""
    graphemes = segment_graphemes("ক্য")
    assert graphemes == ["ক্য"]


def test_grapheme_va_phala_conjunct():
    """ত্ব (ত + ্ + ব) must stay intact as a single grapheme."""
    graphemes = segment_graphemes("ত্ব")
    assert graphemes == ["ত্ব"]


def test_grapheme_complex_kshmi():
    """ক্ষ্মী (ক + ্ + ষ + ্ + ম + ী) must stay intact as a single grapheme."""
    graphemes = segment_graphemes("ক্ষ্মী")
    assert graphemes == ["ক্ষ্মী"]


def test_grapheme_jnya_conjunct():
    """জ্ঞ (জ + ্ + ঞ) stays intact."""
    graphemes = segment_graphemes("জ্ঞান")
    assert graphemes == ["জ্ঞা", "ন"]


def test_grapheme_candrabindu_attached():
    """ঁ (Candrabindu) must stay attached to its base consonant/vowel."""
    graphemes = segment_graphemes("চাঁদ")
    assert graphemes == ["চাঁ", "দ"]


def test_grapheme_anusvara_attached():
    """ং (Anusvara) must stay attached to the preceding syllable."""
    graphemes = segment_graphemes("বাংলা")
    assert graphemes == ["বাং", "লা"]


def test_grapheme_visarga_attached():
    """ঃ (Visarga) must stay attached to the preceding syllable."""
    graphemes = segment_graphemes("দুঃখ")
    assert graphemes == ["দুঃ", "খ"]


def test_grapheme_khanda_ta():
    """ৎ (Khanda Ta) is recognized as its own segment."""
    graphemes = segment_graphemes("উৎসব")
    assert graphemes == ["উ", "ৎ", "স", "ব"]


def test_grapheme_matras_attached():
    """Vowel signs (matras) stay attached to the base consonant."""
    graphemes = segment_graphemes("বৃষ্টি")
    assert graphemes == ["বৃ", "ষ্টি"]


# =========================================================================
# 2. Canonical Forms: য়, ড়, ঢ় variants
# =========================================================================

def test_canonical_yyo_decomposed():
    """Decomposed য + ় (\u09AF\u09BC) normalizes to canonical য় (\u09DF)."""
    decomposed = "\u09AF\u09BC"
    normalized = normalize(decomposed)
    assert normalized == "\u09DF"


def test_canonical_yyo_in_word():
    """Word with decomposed য + ় normalizes to canonical য়."""
    word_decomp = "সম\u09AF\u09BC"  # সময় with decomposed য়
    assert normalize(word_decomp) == "সম\u09DF"


def test_canonical_rro_decomposed():
    """Decomposed ড + ় (\u09A1\u09BC) normalizes to canonical ড় (\u09DC)."""
    decomposed = "\u09A1\u09BC"
    normalized = normalize(decomposed)
    assert normalized == "\u09DC"


def test_canonical_rro_in_word():
    """Word with decomposed ড + ় normalizes to canonical ড়."""
    word_decomp = "বা\u09A1\u09BCি"  # বাড়ি with decomposed ড়
    assert normalize(word_decomp) == "বা\u09DCি"


def test_canonical_rho_decomposed():
    """Decomposed ঢ + ় (\u09A2\u09BC) normalizes to canonical ঢ় (\u09DD)."""
    decomposed = "\u09A2\u09BC"
    normalized = normalize(decomposed)
    assert normalized == "\u09DD"


def test_canonical_rho_in_word():
    """Word with decomposed ঢ + ় normalizes to canonical ঢ়."""
    word_decomp = "আষা\u09A2\u09BC"  # আষাঢ় with decomposed ঢ়
    assert normalize(word_decomp) == "আষা\u09DD"


# =========================================================================
# 3. Two-Part Vowel Signs: ো / ৌ
# =========================================================================

def test_two_part_vowel_o_composition():
    """Split ে + া (\u09C7\u09BE) composes to canonical ো (\u09CB)."""
    split_form = "ল\u09C7\u09BEক"  # লোক typed as ে + া
    normalized = normalize(split_form)
    assert normalized == "লোক"
    assert "\u09CB" in normalized


def test_two_part_vowel_au_composition():
    """Split ে + ৗ (\u09C7\u09D7) composes to canonical ৌ (\u09CC)."""
    split_form = "ন\u09C7\u09D7কা"  # নৌকা typed as ে + ৗ
    normalized = normalize(split_form)
    assert normalized == "নৌকা"
    assert "\u09CC" in normalized


def test_two_part_standalone_mapping():
    """Direct mapping test for compose_two_part_vowels."""
    assert compose_two_part_vowels("\u09C7\u09BE") == "\u09CB"
    assert compose_two_part_vowels("\u09C7\u09D7") == "\u09CC"


# =========================================================================
# 4. Zero Width Characters: ZWNJ & ZWJ Policy
# =========================================================================

def test_zwnj_stripped_by_default():
    """ZWNJ (\u200C) is stripped by default to ensure robust retrieval matching."""
    with_zwnj = f"ক{ZWNJ}ক"
    without_zwnj = "কক"
    assert normalize(with_zwnj) == without_zwnj


def test_zwnj_preserved_when_requested():
    """ZWNJ is retained when preserve_formatting=True."""
    with_zwnj = f"ক{ZWNJ}ক"
    assert normalize(with_zwnj, preserve_formatting=True) == with_zwnj


def test_zwj_stripped_by_default():
    """ZWJ (\u200D) is stripped by default for consistent search tokens."""
    with_zwj = f"ক্{ZWJ}ক"
    without_zwj = "ক্ক"
    assert normalize(with_zwj) == without_zwj


def test_zwj_preserved_when_requested():
    """ZWJ is preserved when preserve_formatting=True."""
    with_zwj = f"ক্{ZWJ}"
    assert normalize(with_zwj, preserve_formatting=True) == with_zwj


# =========================================================================
# 5. Bangla Digits to ASCII
# =========================================================================

def test_bangla_digits_to_ascii():
    """All Bengali numerals 0-9 are converted to ASCII numerals."""
    bangla_num = "০১২৩৪৫৬৭৮৯"
    assert convert_digits_to_ascii(bangla_num) == "0123456789"
    assert normalize(bangla_num) == "0123456789"


def test_bangla_year_in_sentence():
    """Bengali year digits in text are normalized to ASCII."""
    text = "১৯৭১ সালের মুক্তিযুদ্ধ"
    assert normalize(text) == "1971 সালের মুক্তিযুদ্ধ"


def test_digits_preserve_flag():
    """When digits_to_ascii=False, Bengali digits remain unchanged."""
    text = "১০টি বই"
    assert normalize(text, digits_to_ascii=False) == "১০টি বই"


# =========================================================================
# 6. Mis-Ordered Pre-Base Vowel Signs Repair (িশক্ষা -> শিক্ষা, etc.)
# =========================================================================

def test_misordered_vowel_shikkha():
    """Repair pre-base vowel: 'িশক্ষা' -> 'শিক্ষা'."""
    misordered = "িশক্ষা"
    assert repair_misordered_vowels(misordered) == "শিক্ষা"
    assert normalize(misordered) == "শিক্ষা"


def test_misordered_vowel_khela():
    """Repair pre-base vowel: 'েখলা' -> 'খেলা'."""
    misordered = "েখলা"
    assert repair_misordered_vowels(misordered) == "খেলা"
    assert normalize(misordered) == "খেলা"


def test_misordered_vowel_toiri():
    """Repair pre-base vowel: 'ৈতরি' -> 'তৈরি'."""
    misordered = "ৈতরি"
    assert repair_misordered_vowels(misordered) == "তৈরি"
    assert normalize(misordered) == "তৈরি"


def test_misordered_vowel_with_conjunct():
    """Repair pre-base vowel before a conjunct: 'েপ্রম' -> 'প্রেম'."""
    misordered = "েপ্রম"
    assert repair_misordered_vowels(misordered) == "প্রেম"
    assert normalize(misordered) == "প্রেম"


def test_misordered_vowel_in_sentence():
    """Repair pre-base vowel in mid-sentence after space."""
    text = "আমাদের িশক্ষা ব্যবস্থা এবং েখলাধুলা"
    repaired = normalize(text)
    assert repaired == "আমাদের শিক্ষা ব্যবস্থা এবং খেলাধুলা"


# =========================================================================
# 7. Repeated Hasanta & Danda Sentence Boundaries
# =========================================================================

def test_repeated_hasanta_collapsed():
    """Multiple consecutive hasantas are collapsed to a single hasanta."""
    repeated = "ক\u09CD\u09CDষ"  # ক্্ষ
    assert normalize(repeated) == "ক্ষ"


def test_double_danda_standardization():
    """Double danda (॥) is standardized to danda (।)."""
    text = "প্রথম বাক্য॥ দ্বিতীয় বাক্য"
    assert normalize(text) == normalize("প্রথম বাক্য। দ্বিতীয় বাক্য")


# =========================================================================
# 8. Tokenizer: Conjunct Preservation, Latin Lowercasing, Hyphen & Apostrophe
# =========================================================================

def test_tokenize_never_splits_khiyo():
    """Tokenizer never splits ক্ষ into separate tokens."""
    tokens = tokenize("ক্ষত্রিয় যোদ্ধা")
    assert tokens == [normalize("ক্ষত্রিয়"), normalize("যোদ্ধা")]


def test_tokenize_never_splits_skro():
    """Tokenizer preserves triple conjunct স্ক্র."""
    tokens = tokenize("স্ক্রিপ্ট তৈরি করো")
    assert tokens == [normalize("স্ক্রিপ্ট"), normalize("তৈরি"), normalize("করো")]


def test_tokenize_never_splits_ro_reph():
    """Tokenizer preserves reph conjunct র্ক."""
    tokens = tokenize("তর্ক বিতর্ক")
    assert tokens == [normalize("তর্ক"), normalize("বিতর্ক")]


def test_tokenize_never_splits_ya_phala():
    """Tokenizer preserves ya-phala ক্য."""
    tokens = tokenize("বাক্য রচনা")
    assert tokens == [normalize("বাক্য"), normalize("রচনা")]


def test_tokenize_never_splits_va_phala():
    """Tokenizer preserves va-phala ত্ব."""
    tokens = tokenize("গুরুত্ব দাও")
    assert tokens == [normalize("গুরুত্ব"), normalize("দাও")]


def test_tokenize_mixed_bangla_english():
    """Mixed Bangla and English tokens with Latin lowercased."""
    text = "Python ৩.১১ এবং বাংলা Search Engine"
    tokens = tokenize(text)
    assert tokens == ["python", "3.11", normalize("এবং"), normalize("বাংলা"), "search", "engine"]


def test_tokenize_intra_word_hyphen_bangla():
    """Intra-word hyphen in Bengali compounds is preserved as a single token."""
    text = "ঢাকা-সিলেট মহাসড়ক"
    tokens = tokenize(text)
    assert tokens == [normalize("ঢাকা-সিলেট"), normalize("মহাসড়ক")]


def test_tokenize_intra_word_hyphen_english():
    """Intra-word hyphen in Latin compounds is preserved."""
    text = "State-of-the-art NLP models and COVID-19"
    tokens = tokenize(text)
    assert tokens == ["state-of-the-art", "nlp", "models", "and", "covid-19"]


def test_tokenize_apostrophe_handling():
    """Apostrophe in English words is preserved and curly apostrophe normalized."""
    text = "Antigravity’s engine and user's query"
    tokens = tokenize(text)
    assert tokens == ["antigravity's", "engine", "and", "user's", "query"]


def test_tokenize_surrounding_quotes_stripped():
    """Surrounding quotation marks are stripped from tokens."""
    text = "তিনি বললেন, 'ঢাকা' একটি \"ঐতিহাসিক\" শহর।"
    tokens = tokenize(text)
    assert tokens == ["তিনি", "বললেন", "ঢাকা", "একটি", "ঐতিহাসিক", "শহর"]


# =========================================================================
# 9. Suffix-Driven Stemmer (rules/suffixes.tsv) Behind a Flag
# =========================================================================

def test_stemmer_gulo_suffix():
    """Stemmer strips গুলো suffix."""
    assert stem_word("বইগুলো", enabled=True) == "বই"


def test_stemmer_guli_suffix():
    """Stemmer strips গুলি suffix."""
    assert stem_word("কলমগুলি", enabled=True) == "কলম"


def test_stemmer_derke_suffix():
    """Stemmer strips compound derke suffix."""
    assert stem_word("মানুষদেরকে", enabled=True) == "মানুষ"


def test_stemmer_der_suffix():
    """Stemmer strips দের suffix."""
    assert stem_word("ছাত্রদের", enabled=True) == "ছাত্র"


def test_stemmer_er_suffix():
    """Stemmer strips ের suffix."""
    assert stem_word("স্কুলের", enabled=True) == "স্কুল"


def test_stemmer_te_suffix():
    """Stemmer strips তে suffix."""
    assert stem_word("ঢাকাতে", enabled=True) == "ঢাকা"


def test_stemmer_ti_and_ta_suffix():
    """Stemmer strips টি and টা suffixes."""
    assert stem_word("বইটি", enabled=True) == "বই"
    assert stem_word("কলমটা", enabled=True) == "কলম"


def test_stemmer_ra_suffix():
    """Stemmer strips রা suffix."""
    assert stem_word("ছেলেরা", enabled=True) == "ছেলে"


def test_stemmer_ke_suffix():
    """Stemmer strips কে suffix when minimum stem length holds."""
    assert stem_word("কাউকে", enabled=True) == "কাউ"


def test_stemmer_protects_short_words():
    """Stemmer protects base words from being over-stripped below min_stem_len."""
    # 'কে' should not be stripped to empty
    assert stem_word("কে", enabled=True) == "কে"
    # 'তে' should not be stripped to empty
    assert stem_word("তে", enabled=True) == "তে"
    # 'কর' should not lose 'র'
    assert stem_word("কর", enabled=True) == "কর"


def test_tokenize_with_stemming_flag():
    """Tokenize function applies stemming only when stem=True."""
    text = "বইগুলো এবং কলমটি"
    unstemmed = tokenize(text, stem=False)
    stemmed = tokenize(text, stem=True)
    assert unstemmed == ["বইগুলো", "এবং", "কলমটি"]
    assert stemmed == ["বই", "এবং", "কলম"]
