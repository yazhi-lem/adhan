"""Unit and regression tests for TextDeduplicator and train/eval overlap filtering.

Covers:
  - Exact duplicate removal
  - Near-duplicate removal (MinHash/LSH) while keeping distinct lines
  - Unicode NFC normalization (composed vs decomposed Tamil vowel signs)
  - Train vs eval overlap filtering and asserting zero overlap
  - Empty input and single-line edge cases
  - Deduplication pipeline CLI / end-to-end execution
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adhan_slm.data.deduplicator import (  # noqa: E402
    TextDeduplicator,
    compute_train_eval_overlap,
    extract_graphemes,
    filter_train_eval_overlap,
    remove_train_eval_overlap,
)
from scripts.run_dedup import run_dedup_pipeline  # noqa: E402


def _docs() -> List[Dict[str, Any]]:
    return [
        {"id": "a", "text": "இது ஒரு சோதனை வாக்கியம்", "source": "wiki"},
        {"id": "b", "text": "இது ஒரு சோதனை வாக்கியம்", "source": "wiki"},  # exact dup of a
        {"id": "c", "text": "முற்றிலும் வேறு உள்ளடக்கம்", "source": "news"},
    ]


def test_deduplicate_returns_real_stats_not_empty_dict() -> None:
    dedup = TextDeduplicator()
    gen, stats = dedup.deduplicate(iter(_docs()))

    kept = list(gen)

    assert [d["id"] for d in kept] == ["a", "c"]

    assert stats != {}
    assert stats["total_seen"] == 3
    assert stats["kept"] == 2
    assert stats["removed"] == 1
    assert stats["exact_duplicates"] == 1
    assert stats["near_duplicates"] == 0
    assert stats["removal_rate"] == 1 / 3
    assert stats["per_source"]["wiki"] == {"seen": 2, "duplicates": 1}
    assert stats["per_source"]["news"] == {"seen": 1, "duplicates": 0}


def test_deduplicate_stats_on_empty_input() -> None:
    dedup = TextDeduplicator()
    empty_docs: List[Dict[str, Any]] = []
    gen, stats = dedup.deduplicate(iter(empty_docs))
    assert list(gen) == []
    assert stats["total_seen"] == 0
    assert stats["removal_rate"] == 0.0


def test_exact_duplicates_removed() -> None:
    """Assert exact duplicate Tamil lines are removed."""
    docs: List[Dict[str, Any]] = [
        {"id": "1", "text": "தமிழ் வாழ்க வளர்க"},
        {"id": "2", "text": "தமிழ் வாழ்க வளர்க"},  # exact dup
        {"id": "3", "text": "செம்மொழித் தமிழ் வெல்க"},
        {"id": "4", "text": "தமிழ் வாழ்க வளர்க"},  # exact dup
    ]
    dedup = TextDeduplicator()
    gen, stats = dedup.deduplicate(docs, check_near=False)
    kept = list(gen)

    assert len(kept) == 2
    assert [d["id"] for d in kept] == ["1", "3"]
    assert stats["total_seen"] == 4
    assert stats["kept"] == 2
    assert stats["exact_duplicates"] == 2
    assert stats["near_duplicates"] == 0


def test_near_duplicates_removed_different_kept() -> None:
    """Assert near-duplicate lines are removed and clearly different lines are kept."""
    # Long sentences with suffix variation -> Jaccard similarity ~80%
    line1 = "இயற்கை எழில் கொஞ்சும் அழகிய மலைக்காடு தமிழ்நாட்டில் மேற்கு தொடர்ச்சி மலையில் அமைந்துள்ளது"
    line2 = "இயற்கை எழில் கொஞ்சும் அழகிய மலைக்காடு தமிழ்நாட்டில் மேற்கு தொடர்ச்சி மலையில் அமைந்திருக்கிறது"
    # Completely different sentence
    line3 = "வானூர்தி பொறியியல் மற்றும் விண்வெளி ஆராய்ச்சி தொழில்நுட்ப வளர்ச்சிக்கு உதவுகிறது"

    docs: List[Dict[str, Any]] = [
        {"id": "1", "text": line1},
        {"id": "2", "text": line2},  # near dup of line1
        {"id": "3", "text": line3},  # completely distinct
    ]

    dedup = TextDeduplicator(threshold=0.75, shingle_size=5)
    gen, stats = dedup.deduplicate(docs, check_near=True)
    kept = list(gen)

    assert len(kept) == 2
    assert [d["id"] for d in kept] == ["1", "3"]
    assert stats["total_seen"] == 3
    assert stats["kept"] == 2
    assert stats["exact_duplicates"] == 0
    assert stats["near_duplicates"] == 1


def test_nfc_composed_vs_decomposed_treated_as_same_line() -> None:
    """Assert composed vs decomposed Tamil vowel signs (e.g. ொ) are treated as identical."""
    # Composed vowel sign ொ: U+0BCA
    composed_text = "கொடி கட்டி பறந்த சோழ பேரரசு"
    # Decomposed: base க (U+0B95) + ெ (U+0BC6) + ா (U+0BBE)
    decomposed_text = "க\u0bc6\u0bbeடி கட்டி பறந்த சோழ பேரரசு"

    # Verify they look the same visually but are different codepoint representations before normalization
    assert composed_text != decomposed_text
    assert unicodedata.normalize("NFC", decomposed_text) == composed_text

    # Both must normalize to identical text
    assert TextDeduplicator.normalize_text(composed_text) == TextDeduplicator.normalize_text(
        decomposed_text
    )

    docs: List[Dict[str, Any]] = [
        {"id": "comp", "text": composed_text},
        {"id": "decomp", "text": decomposed_text},
    ]

    # Exact dedup pass should identify decomposed as exact duplicate of composed
    dedup = TextDeduplicator()
    gen, stats = dedup.deduplicate(docs, check_near=False)
    kept = list(gen)

    assert len(kept) == 1
    assert kept[0]["id"] == "comp"
    assert stats["exact_duplicates"] == 1
    assert stats["kept"] == 1


def test_grapheme_extraction_preserves_tamil_clusters() -> None:
    """Assert extract_graphemes treats Tamil aksharas and combining marks as atomic."""
    clusters = extract_graphemes("தமிழ் நாடு")
    # 'த', 'மி' (ம+ி), 'ழ்' (ழ+்), ' ', 'நா' (ந+ா), 'டு' (ட+ு)
    assert clusters == ["த", "மி", "ழ்", " ", "நா", "டு"]


def test_train_eval_overlap_removed_and_zero_afterward() -> None:
    """Assert train lines appearing in eval set are removed, and final overlap == 0."""
    train_lines = [
        "பண்டைத் தமிழகத்தில் சங்க இலக்கியங்கள் தோன்றின",
        "கரிகால் சோழன் கல்லணையைக் கட்டினான்",  # LEAK: in eval
        "மதுரை மாநகர் தமிழ் வளர்த்த கூடல் நகரம்",
        "க\u0bc6\u0bbeடி கட்டி பறந்த சோழ பேரரசு",  # LEAK with decomposed ொ: in eval as composed
    ]

    eval_lines = [
        "கரிகால் சோழன் கல்லணையைக் கட்டினான்",
        "கொடி கட்டி பறந்த சோழ பேரரசு",  # composed ொ
        "சிலப்பதிகாரம் கண்ணகியின் வரலாற்றைக் கூறுகிறது",
    ]

    # Pre-condition: overlap is 2
    initial_overlap = compute_train_eval_overlap(train_lines, eval_lines)
    assert initial_overlap == 2

    # Filter out overlap
    filtered_train, removed_count = filter_train_eval_overlap(train_lines, eval_lines)

    assert removed_count == 2
    assert len(filtered_train) == 2
    assert filtered_train == [
        "பண்டைத் தமிழகத்தில் சங்க இலக்கியங்கள் தோன்றின",
        "மதுரை மாநகர் தமிழ் வளர்த்த கூடல் நகரம்",
    ]

    # Post-condition: overlap MUST be zero
    post_overlap = compute_train_eval_overlap(filtered_train, eval_lines)
    assert post_overlap == 0

    # In-place version
    train_copy = list(train_lines)
    removed_inplace = remove_train_eval_overlap(train_copy, eval_lines)
    assert removed_inplace == 2
    assert train_copy == filtered_train
    assert compute_train_eval_overlap(train_copy, eval_lines) == 0


def test_empty_and_single_line_input_handled() -> None:
    """Assert empty input and single-line input are handled gracefully across all components."""
    dedup = TextDeduplicator()

    # Empty input
    empty_list: List[str] = []
    filtered, removed = filter_train_eval_overlap(empty_list, ["சில வாசகம்"])
    assert filtered == []
    assert removed == 0
    assert compute_train_eval_overlap(empty_list, ["சில வாசகம்"]) == 0

    # Single-line input: no match in eval
    single_train = ["ஒற்றை வரி சோதனை வாக்கியம்"]
    filtered_single, rem_single = filter_train_eval_overlap(single_train, ["வேறு வாசகம்"])
    assert filtered_single == single_train
    assert rem_single == 0
    assert compute_train_eval_overlap(filtered_single, ["வேறு வாசகம்"]) == 0

    # Single-line input: exact match in eval
    filtered_leak, rem_leak = filter_train_eval_overlap(single_train, single_train)
    assert filtered_leak == []
    assert rem_leak == 1
    assert compute_train_eval_overlap(filtered_leak, single_train) == 0

    # Single-line dedup
    gen, stats = dedup.deduplicate([{"id": "only", "text": "ஒற்றை வரி"}])
    kept = list(gen)
    assert len(kept) == 1
    assert stats["kept"] == 1
    assert stats["removed"] == 0


def test_cli_pipeline_run_dedup_end_to_end(tmp_path: Path) -> None:
    """Test full CLI pipeline: exact dedup -> near dedup -> train/eval filter."""
    train_file = tmp_path / "train.txt"
    eval_file = tmp_path / "eval.txt"
    output_file = tmp_path / "train_clean.txt"
    report_file = tmp_path / "report.json"

    # Synthetic fixtures; need native Tamil review per #70 definition of done.
    # Train corpus:
    # Line 1: base line
    # Line 2: exact dup of line 1
    # Line 3: near dup of line 1 (1 word change)
    # Line 4: eval leak line
    # Line 5: unique line
    train_content = "\n".join(
        [
            "இயற்கை எழில் கொஞ்சும் அழகிய மலைக்காடு தமிழ்நாட்டில் மேற்கு தொடர்ச்சி மலையில் அமைந்துள்ளது",
            "இயற்கை எழில் கொஞ்சும் அழகிய மலைக்காடு தமிழ்நாட்டில் மேற்கு தொடர்ச்சி மலையில் அமைந்துள்ளது",  # exact dup
            "இயற்கை எழில் கொஞ்சும் அழகிய மலைக்காடு தமிழ்நாட்டில் மேற்கு தொடர்ச்சி மலையில் அமைந்திருக்கிறது",  # near dup (~80% sim)
            "வைகை ஆற்றின் கரையில் மக்கள் மாலை நேரத்தில் நடைபயிற்சி செய்கிறார்கள்",  # eval leak
            "அறிவே ஆற்றல் என்பது வள்ளுவர் வாக்கு போன்ற பழமொழி",  # unique
        ]
    )
    train_file.write_text(train_content, encoding="utf-8")

    eval_content = "\n".join(
        [
            "வைகை ஆற்றின் கரையில் மக்கள் மாலை நேரத்தில் நடைபயிற்சி செய்கிறார்கள்",
            "ஞாயிற்றுக்கிழமை சந்தையில் காய்கறி மற்றும் பழங்கள் விற்பனை விறுவிறுப்பாக நடக்கிறது",
        ]
    )
    eval_file.write_text(eval_content, encoding="utf-8")

    report = run_dedup_pipeline(
        corpus_path=train_file,
        eval_corpus_path=eval_file,
        output_path=output_file,
        report_path=report_file,
        threshold=0.75,
    )

    # Validate counts
    assert report["stages"]["initial_count"] == 5
    assert report["stages"]["exact_dedup"]["removed"] == 1
    assert report["stages"]["exact_dedup"]["after"] == 4
    assert report["stages"]["near_dedup"]["removed"] == 1
    assert report["stages"]["near_dedup"]["after"] == 3
    assert report["stages"]["train_eval_filter"]["removed"] == 1
    assert report["stages"]["train_eval_filter"]["after"] == 2
    assert report["stages"]["train_eval_filter"]["final_eval_overlap"] == 0
    assert report["final_count"] == 2
    assert report["total_removed"] == 3

    # Validate output file
    assert output_file.exists()
    out_lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(out_lines) == 2

    # Validate JSON report file
    assert report_file.exists()
    saved_report = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved_report["final_count"] == 2
    assert saved_report["stages"]["train_eval_filter"]["final_eval_overlap"] == 0


def test_per_source_dedup() -> None:
    """Assert per_source_dedup removes duplicates within each source independently."""
    dedup = TextDeduplicator()
    docs = [
        {"id": "w1", "text": "தமிழ்", "source": "wiki"},
        {"id": "w2", "text": "தமிழ்", "source": "wiki"},  # dup within wiki
        {"id": "n1", "text": "தமிழ்", "source": "news"},  # first in news -> kept
        {"id": "n2", "text": "தமிழ்", "source": "news"},  # dup within news
    ]
    kept = list(dedup.per_source_dedup(docs))
    assert [d["id"] for d in kept] == ["w1", "n1"]


def test_write_stats(tmp_path: Path) -> None:
    """Assert write_stats outputs valid JSON."""
    dedup = TextDeduplicator()
    stats = {"total_seen": 10, "kept": 8, "removed": 2}
    out_file = tmp_path / "stats.json"
    dedup.write_stats(stats, out_file)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded == stats


def test_deduplicator_class_static_methods() -> None:
    """Assert TextDeduplicator static methods delegate correctly."""
    train = ["வரி ஒன்று", "வரி இரண்டு"]
    eval_set = ["வரி இரண்டு"]

    filtered, removed = TextDeduplicator.filter_train_eval_overlap(train, eval_set)
    assert filtered == ["வரி ஒன்று"]
    assert removed == 1

    overlap = TextDeduplicator.compute_train_eval_overlap(filtered, eval_set)
    assert overlap == 0

    train_mut = list(train)
    rem = TextDeduplicator.remove_train_eval_overlap(train_mut, eval_set)
    assert rem == 1
    assert train_mut == ["வரி ஒன்று"]

