"""Tests for the data pipeline: markup cleaner, chunker, and dump stream-parser."""

import bz2
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from bsearch.data.chunker import chunk_article, split_sentences
from bsearch.data.dump_parser import parse_and_export_dump
from bsearch.data.markup_cleaner import clean_wiki_markup, clean_wikilinks, strip_templates


def test_strip_templates():
    raw = "ভূমিকা {{তথ্যছক শহর | নাম = ঢাকা | দেশ = বাংলাদেশ }} ঢাকা বাংলাদেশের রাজধানী।"
    cleaned = strip_templates(raw)
    assert "তথ্যছক" not in cleaned
    assert "ঢাকা বাংলাদেশের রাজধানী।" in cleaned


def test_clean_wikilinks():
    raw = "এখানে [[বাংলাদেশ]] এবং [[গণপ্রজাতন্ত্রী বাংলাদেশ|বাংলাদেশ রাষ্ট্র]] রয়েছে। [[চিত্র:Dhaka.jpg|ছবি]]"
    cleaned = clean_wikilinks(raw)
    assert "বাংলাদেশ" in cleaned
    assert "বাংলাদেশ রাষ্ট্র" in cleaned
    assert "চিত্র:Dhaka.jpg" not in cleaned


def test_clean_wiki_markup_comprehensive():
    wikitext = """
<!-- এই একটি মন্তব্য -->
{{তথ্যছক ব্যক্তি
| নাম = রবীন্দ্রনাথ ঠাকুর
}}
'''রবীন্দ্রনাথ ঠাকুর''' (৭ মে ১৮৬১ – ৭ আগস্ট ১৯৪১) ছিলেন একজন [[বাঙালি]] [[কবি]] ও সাহিত্যিক।<ref>উৎস উদ্ধৃতি</ref>

== জীবনী ==
{| class="wikitable"
|-
! সাল !! ঘটনা
|-
| ১৮৬১ || জন্ম
|}
তিনি [[গীতাঞ্জলি|গীতাঞ্জলি কাব্যগ্রন্থের]] জন্য নোবেল পুরস্কার লাভ করেন। [https://example.com সরকারি তথ্যসূত্র]

[[বিষয়শ্রেণী:বাঙালি কবি]]
[[চিত্র:Rabindranath_Tagore.jpg|thumb|রবীন্দ্রনাথ ঠাকুর]]
"""
    cleaned = clean_wiki_markup(wikitext)

    # Verifications
    assert "মন্তব্য" not in cleaned
    assert "তথ্যছক ব্যক্তি" not in cleaned
    assert "উৎস উদ্ধৃতি" not in cleaned
    assert "wikitable" not in cleaned
    assert "বিষয়শ্রেণী" not in cleaned
    assert "চিত্র:Rabindranath" not in cleaned
    assert "https://example.com" not in cleaned
    assert "রবীন্দ্রনাথ ঠাকুর" in cleaned
    assert "বাঙালি" in cleaned
    assert "গীতাঞ্জলি কাব্যগ্রন্থের" in cleaned
    assert "নোবেল পুরস্কার লাভ করেন" in cleaned
    assert "জীবনী" in cleaned


def test_split_sentences():
    text = "ঢাকা বাংলাদেশের রাজধানী। এটি বুড়িগঙ্গা নদীর তীরে অবস্থিত? হ্যাঁ, এটি অত্যন্ত প্রাচীন নগরী! এখানে বহু মানুষ বাস করে।"
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "ঢাকা বাংলাদেশের রাজধানী।"
    assert sentences[1] == "এটি বুড়িগঙ্গা নদীর তীরে অবস্থিত?"
    assert sentences[2] == "হ্যাঁ, এটি অত্যন্ত প্রাচীন নগরী!"
    assert sentences[3] == "এখানে বহু মানুষ বাস করে।"


def test_chunk_article_small():
    article_id = "101"
    title = "ছোট নিবন্ধ"
    # 40 words
    text = " ".join(["শব্দ"] * 40) + "।"
    chunks = chunk_article(article_id, title, text, target_words=150, min_words=20)
    assert len(chunks) == 1
    assert chunks[0]["article_id"] == "101"
    assert chunks[0]["title"] == "ছোট নিবন্ধ"
    assert chunks[0]["word_count"] == 40
    assert "CC BY-SA 4.0" in chunks[0]["source"]


def test_chunk_article_large():
    article_id = "202"
    title = "দীর্ঘ নিবন্ধ"
    # Create text with 350 words divided into sentences of 25 words
    sentences = [" ".join(["শব্দ"] * 24) + "।" for _ in range(14)]
    text = " ".join(sentences)

    chunks = chunk_article(
        article_id, title, text, target_words=150, overlap_words=25, min_words=30
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["article_id"] == "202"
        assert chunk["title"] == "দীর্ঘ নিবন্ধ"
        assert chunk["word_count"] >= 30
        assert "CC BY-SA 4.0" in chunk["source"]


def test_dump_stream_parser(tmp_path: Path):
    # Construct a mini Wikipedia XML dump
    xml_content = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>Wikipedia:নীতিমালা</title>
    <ns>4</ns>
    <id>1</id>
    <revision>
      <id>10</id>
      <text>এটি নীতিমালার পাতা যা নন-মেইনস্পেস।</text>
    </revision>
  </page>
  <page>
    <title>ঢাকা শহর</title>
    <ns>0</ns>
    <id>2</id>
    <redirect title="ঢাকা" />
    <revision>
      <id>20</id>
      <text>#পুনর্নির্দেশ [[ঢাকা]]</text>
    </revision>
  </page>
  <page>
    <title>ছোট পাতা</title>
    <ns>0</ns>
    <id>3</id>
    <revision>
      <id>30</id>
      <text>অল্প লেখা।</text>
    </revision>
  </page>
  <page>
    <title>দ্ব্যর্থতা পাতা</title>
    <ns>0</ns>
    <id>4</id>
    <revision>
      <id>40</id>
      <text>{{দ্ব্যর্থতা নিরসন}} এটি একাধিক বিষয়ের জন্য ব্যবহৃত হয়।</text>
    </revision>
  </page>
  <page>
    <title>বাংলাদেশ</title>
    <ns>0</ns>
    <id>5</id>
    <revision>
      <id>50</id>
      <text>'''বাংলাদেশ''' দক্ষিণ এশিয়ার একটি স্বাধীন ও সার্বভৌম রাষ্ট্র। এর রাজধানী ঢাকা। পদ্মা, মেঘনা ও যমুনা এর প্রধান নদী। এটি প্রাকৃতিক সৌন্দর্যে ভরপুর একটি দেশ। এখানকার মানুষ বাংলা ভাষায় কথা বলে এবং সংস্কৃতি অত্যন্ত সমৃদ্ধ। এই দেশে বহু প্রাচীন ঐতিহ্য রয়েছে।</text>
    </revision>
  </page>
</mediawiki>"""

    dump_xml_path = tmp_path / "test_bnwiki.xml"
    dump_xml_path.write_text(xml_content, encoding="utf-8")

    # Also test bz2 compression handling
    dump_bz2_path = tmp_path / "test_bnwiki.xml.bz2"
    with bz2.open(dump_bz2_path, "wb") as f_bz2:
        f_bz2.write(xml_content.encode("utf-8"))

    art_out = tmp_path / "articles.jsonl"
    chunk_out = tmp_path / "passages.jsonl"

    stats = parse_and_export_dump(
        dump_path=dump_bz2_path,
        article_output_path=art_out,
        chunk_output_path=chunk_out,
        min_chars=30,
        min_words=10,
        force_chunk=True,
    )

    assert stats["total_pages_seen"] == 5
    assert stats["non_mainspace_skipped"] == 1
    assert stats["redirects_skipped"] == 1
    assert stats["disambiguations_skipped"] == 1
    assert stats["stubs_skipped"] == 1
    assert stats["valid_articles"] == 1
    assert stats["passages_created"] >= 1

    # Inspect generated JSONL
    assert art_out.exists()
    lines = [json.loads(line) for line in art_out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    article = lines[0]
    assert article["id"] == "5"
    assert article["title"] == "বাংলাদেশ"
    assert "দক্ষিণ এশিয়ার একটি স্বাধীন ও সার্বভৌম রাষ্ট্র" in article["text"]
    assert "CC BY-SA 4.0" in article["source"]

    # Inspect generated passages
    assert chunk_out.exists()
    chunk_lines = [
        json.loads(line) for line in chunk_out.read_text(encoding="utf-8").splitlines()
    ]
    assert len(chunk_lines) >= 1
    assert chunk_lines[0]["article_id"] == "5"
    assert chunk_lines[0]["title"] == "বাংলাদেশ"
