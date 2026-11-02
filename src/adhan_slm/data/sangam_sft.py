"""Build Sangam supervision from the labelled corpus (verse + urai + tiṇai).

The Sangam corpus is tiny (~0.5M tokens) and its value is structure, not volume:
which landscape, which season, which emotion, which imagery. This module turns the
already-labelled records into deterministic SFT tasks and a held-out classification
benchmark. No teacher model is involved, so there is nothing to hallucinate in the
inputs; a wrong target would be a labelling bug, and the labels come from
``docs/corpus/SANGAM_INGEST.md``'s canon.

Splits are **by poem**, keyed by a stable hash of the poem id: every task generated
from one poem lands in exactly one split, so a test verse never appears in training
in another task form.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from adhan_slm import protocol as p
from adhan_slm.data.sangam_tinai import canonical_tinai, load_canon

#: Romanised corpus work id -> canonical Tamil work name used in citations.
WORK_TAMIL: Dict[str, str] = {
    "ainkurunooru": "ஐங்குறுநூறு",
    "akananooru": "அகநானூறு",
    "kalithokai": "கலித்தொகை",
    "kurinjipattu": "குறிஞ்சிப்பாட்டு",
    "kurunthokai": "குறுந்தொகை",
    "maduraikanchi": "மதுரைக்காஞ்சி",
    "malaipadukadam": "மலைபடுகடாம்",
    "mullaippattu": "முல்லைப்பாட்டு",
    "natrinai": "நற்றிணை",
    "nedunalvadai": "நெடுநல்வாடை",
    "paripadal": "பரிபாடல்",
    "pathitrupathu": "பதிற்றுப்பத்து",
    "pattinappalai": "பட்டினப்பாலை",
    "perumpanatrupadai": "பெரும்பாணாற்றுப்படை",
    "porunaratrupadai": "பொருநராற்றுப்படை",
    "purananooru": "புறநானூறு",
    "sirupanatrupadai": "சிறுபாணாற்றுப்படை",
    "thirumurugatrupadai": "திருமுருகாற்றுப்படை",
}

SYSTEM = "நீ ஒரு தமிழ் இலக்கிய உதவியாளர். சங்க இலக்கியப் பாடல்களைத் தமிழில் விளக்குவாய்."
SYSTEM_EN = (
    "நீ ஒரு தமிழ் இலக்கிய மொழிபெயர்ப்பாளர். "
    "சங்கப் பாடல்களை இயல்பான ஆங்கிலத்தில் மொழிபெயர்ப்பாய்."
)

TASK_TINAI = "sangam_tinai"
TASK_URAI = "sangam_urai"
TASK_CITATION = "sangam_citation"
TASK_KARUPPORUL = "sangam_karupporul"
TASK_TINAI_FROM_URAI = "sangam_tinai_from_urai"
TASK_CITATION_FROM_URAI = "sangam_citation_from_urai"
TASK_EURAI_EN = "sangam_eurai_en"


def work_tamil(work: str) -> str:
    return WORK_TAMIL.get(work, work)


def reference_of(rec: dict) -> str:
    """Citation locator: the labelled number, else the id suffix.

    Some works (e.g. thirumurugatrupadai) have no ``number`` in the labels but do
    have ``work_01``-style ids. Falling back to the suffix is better than emitting
    the literal string ``None`` inside a citation.
    """
    if rec.get("number") is not None:
        return str(rec["number"])
    return str(rec["poem_id"]).rsplit("_", 1)[-1]


def split_of(poem_id: str, val_frac: float = 0.10, test_frac: float = 0.10) -> str:
    """Deterministic poem-level split. Boundaries are validated by the caller."""
    digest = hashlib.sha1(poem_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    bucket = int(digest[:8], 16) % 10_000 / 10_000.0
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_open_sangam(open_sangam_root: str | Path | None) -> Dict[str, dict]:
    """Per-poem ``sangamTamil`` / ``urai`` / ``english`` from Open-Sangam.

    The adhan ingest gives the verse/urai **text** in `data/raw/sangam`, but it
    covers fewer poems (674 verse texts) and deliberately excludes the ``english``
    field (it is not Tamil corpus text). Open-Sangam's source records are therefore
    used as a *fallback* for missing verse/urai, and as the only source of the
    English rendering for the ``sangam_eurai_en`` task.

    English coverage is currently partial. Fill the rest with the Open-Sangam
    pipeline, then rerun this builder; no code change is needed as coverage grows::

        cd ~/Yazhi/repos/open-sangam/scripts
        OPENROUTER_API_KEY=... python -m ai.translate_all --only english
    """
    if not open_sangam_root:
        return {}
    texts = Path(open_sangam_root) / "data" / "texts"
    if not texts.exists():
        return {}
    out: Dict[str, dict] = {}
    for path in sorted(texts.glob("*/*.json")):
        if path.parent.name == "normalized" or path.name == "datapackage.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            pid = rec.get("id")
            if not pid:
                continue
            out[str(pid)] = {
                "verse": (rec.get("sangamTamil") or "").strip() or None,
                "urai": (rec.get("urai") or "").strip() or None,
                "english": (rec.get("english") or "").strip() or None,
            }
    return out


def load_records(
    labels_path: str | Path,
    sangam_dir: str | Path,
    open_sangam_root: str | Path | None = None,
) -> List[dict]:
    """Join labels with the verse and (traditional) urai text, keyed by ``poem_id``.

    ``yazhi-urai`` is model-generated commentary and is deliberately excluded: it is
    useful for comparison, not as a gold target. A record is kept when it has *either*
    a verse or a traditional urai, because the urai-only poems still support the
    landscape/citation/imagery tasks (1,847 urai vs only 674 verse texts).
    """
    labels = list(_read_jsonl(Path(labels_path)))
    open_recs = load_open_sangam(open_sangam_root)
    verses: Dict[str, dict] = {}
    urais: Dict[str, dict] = {}
    for path in sorted(Path(sangam_dir).glob("*/verse.jsonl")):
        for rec in _read_jsonl(path):
            pid = rec.get("poem_id")
            text = (rec.get("text") or "").strip()
            if pid and text:
                verses[pid] = rec
    for path in sorted(Path(sangam_dir).glob("*/urai.jsonl")):
        for rec in _read_jsonl(path):
            pid = rec.get("poem_id")
            text = (rec.get("text") or "").strip()
            if pid and text:
                urais[pid] = rec

    records: List[dict] = []
    seen = set()
    for lab in labels:
        pid = lab.get("id")
        if not pid or pid in seen:
            continue
        verse = verses.get(pid) or {}
        urai = urais.get(pid) or {}
        fallback = open_recs.get(pid) or {}
        verse_text = (verse.get("text") or "").strip() or fallback.get("verse")
        urai_text = (urai.get("text") or "").strip() or fallback.get("urai")
        if not (verse_text or urai_text):
            continue
        seen.add(pid)
        records.append(
            {
                "poem_id": pid,
                "work": lab.get("work") or verse.get("work") or "",
                "number": lab.get("number"),
                "verse": verse_text,
                "urai": urai_text,
                "english": fallback.get("english"),
                "tinai": canonical_tinai(lab.get("tinai")),
                "tinai_tamil": lab.get("tinai_tamil")
                or (
                    load_canon().get(canonical_tinai(lab.get("tinai"))).tamil
                    if canonical_tinai(lab.get("tinai"))
                    else None
                ),
                "kalam": lab.get("kalam"),
                "uripporul_tamil": lab.get("uripporul_tamil"),
                "karupporul": list(lab.get("karupporul") or []),
                "poet": lab.get("poet"),
            }
        )
    records.sort(key=lambda r: r["poem_id"])
    return records


def _prompt(text: str, question: str, label: str = "பாடல்") -> List[dict]:
    return [{"role": "user", "content": f"{label}:\n{text}\n\n{question}"}]


def _stimulus(rec: dict) -> Tuple[str, str]:
    """The text to show and its Tamil label: verse when present, otherwise urai."""
    if rec["verse"]:
        return rec["verse"], "பாடல்"
    return rec["urai"] or "", "விளக்கம்"


def _example(
    rec: dict,
    task: str,
    messages: List[dict],
    answer: str,
    citations: Sequence[str],
    thought: Optional[str] = None,
    system: str = SYSTEM,
) -> dict:
    prompt, target = p.render_example(
        messages, answer, thought=thought, citations=citations, system=system
    )
    return {
        "id": f"{task}:{rec['poem_id']}",
        "task": task,
        "prompt": prompt,
        "target": target,
        "answer": answer,
        "citations": list(citations),
        "messages": messages,
        "meta": {
            "poem_id": rec["poem_id"],
            "work": rec["work"],
            "number": rec["number"],
            "reference": reference_of(rec),
            "tinai": rec["tinai"],
            "tinai_tamil": rec["tinai_tamil"],
        },
    }


def build_examples(records: Sequence[dict]) -> List[dict]:
    """One poem -> several tasks. Deterministic order; no random sampling."""
    out: List[dict] = []
    for rec in records:
        ref = reference_of(rec)
        cite = [p.citation(work_tamil(rec["work"]), ref)]
        verse, urai = rec["verse"], rec["urai"]
        tinai = rec["tinai_tamil"]
        text, label = _stimulus(rec)

        if tinai:
            out.append(
                _example(
                    rec,
                    TASK_TINAI,
                    _prompt(text, "இந்தப் பாடலின் திணை எது? திணையின் பெயரை மட்டும் சொல்.", label),
                    tinai,
                    cite,
                )
            )
            if urai:
                out.append(
                    _example(
                        rec,
                        TASK_TINAI_FROM_URAI,
                        [
                            {
                                "role": "user",
                                "content": f"விளக்கம்:\n{urai}\n\n"
                                "இந்த விளக்கம் எந்தத் திணையைச் சேர்ந்தது? "
                                "திணையின் பெயரை மட்டும் சொல்.",
                            }
                        ],
                        tinai,
                        cite,
                    )
                )

        out.append(
            _example(
                rec,
                TASK_CITATION,
                _prompt(
                    text,
                    "இந்தப் பாடல் எந்த நூலில், எந்தப் பாடலாக உள்ளது? "
                    "நூல் பெயரும் பாடல் எண்ணும் சொல்.",
                    label,
                ),
                f"{work_tamil(rec['work'])} {ref}",
                cite,
            )
        )
        if urai and not verse:
            out.append(
                _example(
                    rec,
                    TASK_CITATION_FROM_URAI,
                    [
                        {
                            "role": "user",
                            "content": f"விளக்கம்:\n{urai}\n\n"
                            "இந்த விளக்கம் எந்த நூலின் எந்தப் பாடலுக்குச் சொல்லப்பட்டது? "
                            "நூல் பெயரும் பாடல் எண்ணும் சொல்.",
                        }
                    ],
                    f"{work_tamil(rec['work'])} {ref}",
                    cite,
                )
            )

        if rec["karupporul"]:
            out.append(
                _example(
                    rec,
                    TASK_KARUPPORUL,
                    _prompt(text, "இந்தப் பாடலின் கருப்பொருள்களைப் பட்டியலிடு.", label),
                    ", ".join(rec["karupporul"]),
                    cite,
                )
            )

        if verse and urai:
            out.append(
                _example(
                    rec,
                    TASK_URAI,
                    _prompt(verse, "இந்தப் பாடலின் பொருளை நவீன தமிழில் விளக்கு."),
                    urai,
                    cite,
                )
            )

        # eurai (English rendering): supervised only where Open-Sangam already has a
        # human/teacher translation. Never machine-translate a *target* inside the
        # builder: a wrong gold translation teaches fluent mistranslation.
        # The Open-Sangam ``english`` field translates the *verse*. Do not pair it
        # with a urai-only record: that would train "translate this commentary"
        # against a verse translation, i.e. a mismatched gold pair.
        if rec.get("english") and verse:
            out.append(
                _example(
                    rec,
                    TASK_EURAI_EN,
                    [
                        {
                            "role": "user",
                            "content": f"பாடல்:\n{verse}\n\n"
                            "இந்தப் பாடலின் ஆங்கில உரை (eurai) யாது? "
                            "ஆங்கிலத்தில் மட்டும் எழுது.",
                        }
                    ],
                    rec["english"],
                    cite,
                    system=SYSTEM_EN,
                )
            )
    return out


def split_examples(records: Sequence[dict]) -> Dict[str, List[dict]]:
    """Assign poems to splits, then build examples per split."""
    by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for rec in records:
        by_split[split_of(rec["poem_id"])].append(rec)
    return {split: build_examples(recs) for split, recs in by_split.items()}


def build_benchmark(records: Sequence[dict], seed: int = 42) -> List[dict]:
    """Held-out tiṇai MCQ rows for ``bench/tamil-bench``.

    The answer is a letter, because that is what an API model can be scored on
    deterministically. Options are the five canonical tiṇais, shuffled per row with a
    fixed seed; rows whose label is unknown are excluded (they would test nothing).
    """
    canon = load_canon()
    names = [canon[k].tamil for k in ("kurinji", "mullai", "marutam", "neytal", "palai")]
    rng = random.Random(seed)
    rows: List[dict] = []
    for rec in records:
        if not rec["tinai"]:
            continue
        text, label = _stimulus(rec)
        if not text:
            continue
        options = list(names)
        rng.shuffle(options)
        rows.append(
            {
                "id": rec["poem_id"],
                "kind": "verse" if rec["verse"] else "urai",
                "context": text,
                "context_label": label,
                "question": "இந்தப் பாடலின் திணை எது?",
                "options": options,
                "answer_index": options.index(rec["tinai_tamil"]),
                "work": rec["work"],
                "work_tamil": work_tamil(rec["work"]),
                "number": rec["number"],
                "tinai": rec["tinai"],
            }
        )
    return rows


def stats(records: Sequence[dict], examples: Dict[str, List[dict]]) -> dict:
    def count(xs: Iterable[dict], key: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for x in xs:
            k = str(x.get(key))
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    def tinai_counts(xs: Iterable[dict]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in xs:
            if e["task"] == TASK_TINAI:
                k = str(e["meta"].get("tinai"))
                out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    poem_splits = {
        split: {r["poem_id"] for r in recs} for split, recs in _records_by_split(records).items()
    }
    overlap = {
        "train_test": len(poem_splits["train"] & poem_splits["test"]),
        "train_val": len(poem_splits["train"] & poem_splits["val"]),
        "val_test": len(poem_splits["val"] & poem_splits["test"]),
    }
    return {
        "poems": len(records),
        "poems_by_split": {k: len(v) for k, v in poem_splits.items()},
        "poem_overlap": overlap,
        "examples_by_split": {k: len(v) for k, v in examples.items()},
        "examples_by_task": count((e for xs in examples.values() for e in xs), "task"),
        "tinai_by_split": {split: tinai_counts(xs) for split, xs in examples.items()},
        "poems_with_verse": sum(1 for r in records if r["verse"]),
        "poems_with_english": sum(1 for r in records if r.get("english")),
        "poems_with_urai": sum(1 for r in records if r["urai"]),
        "poems_with_karupporul": sum(1 for r in records if r["karupporul"]),
        "unknown_tinai": sum(1 for r in records if not r["tinai"]),
    }


def _records_by_split(records: Sequence[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for rec in records:
        out[split_of(rec["poem_id"])].append(rec)
    return out
