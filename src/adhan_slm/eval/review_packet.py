"""Native-review packet and sign-off validation for the frozen eval corpus (adhan#112, K6).

The corpus cannot be published on code alone: a native speaker has to sign off the exclusion
gazetteer and the sampled sentences. That review is only cheap if it arrives as a turnkey
packet and only trustworthy if the sign-off can be *checked* later.

Two artefacts, deliberately different in nature:

* **The packet** (`data/review/*.md`, git-ignored) contains the sampled text, so a reviewer can
  read it offline. It never enters the repository: the Tanglish rows are scraped comments and
  `data/` is git-ignored for exactly this reason.
* **The sign-off** (`eval_manifests/*.json`, committed) contains verdicts keyed by sentence id
  plus a SHA-256 per sentence — never the text. That makes it safe to commit, and it ties the
  review to one exact freeze: if the corpus changes, the hashes stop matching.

Validation is the enforcement point. ``validate_signoff`` returns a list of problems, so a
release step (or CI, later) can fail closed until a human has signed off the exact bytes that
would be published.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

#: Bumped when the sign-off file's shape changes.
SIGNOFF_SCHEMA_VERSION = 1

#: Verdicts a reviewer may record per sentence.
VERDICT_KEEP = "keep"
VERDICT_DROP = "drop"
VERDICT_PENDING = "pending"
VERDICTS = (VERDICT_KEEP, VERDICT_DROP, VERDICT_PENDING)

#: Registers whose sample carries scraped social-media risk.
REVIEW_FOCUS = ("tanglish",)


def text_sha256(text: str) -> str:
    """Hash of the NFC-normalised sentence: the review key that is safe to commit."""
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


def _by_register(records: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(str(rec["register"]), []).append(rec)
    return grouped


def packet_markdown(
    records: Sequence[Mapping[str, Any]],
    corpus_sha256: str,
    filter_desc: Optional[Mapping[str, Any]],
    blocklist_terms: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
    focus: Sequence[str] = REVIEW_FOCUS,
) -> str:
    """Render the reviewer packet. Contains text; write it outside the repository."""
    grouped = _by_register(records)
    flt = dict(filter_desc or {})
    lines = [
        "# Native review packet — frozen tokenizer eval corpus",
        "",
        "Task: [K6 / #112](https://github.com/yazhi-lem/adhan/issues/112). This packet is",
        "generated for local use only. It contains sampled corpus text and must not be",
        "committed or uploaded: the Tanglish rows are scraped comments and the release rules",
        "keep raw text out of the repository.",
        "",
        "## What you are signing off",
        "",
        "1. **The exclusion gazetteer** — does it cover the political, abusive, sexual and",
        "   violent Tamil/Tanglish vocabulary you would expect, and does it avoid blocking",
        "   ordinary speech? Missing terms are a defect; mark them in `missing_terms`.",
        "2. **The sampled sentences** — is each one something this project is willing to",
        "   publish in a public benchmark? Verdict per sentence: `keep` or `drop`.",
        "",
        "## Freeze identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Corpus output SHA-256 | `{corpus_sha256}` |",
        f"| Sentence count | {len(records):,} |",
        f"| Filter version | `{flt.get('version', 'none')}` |",
        f"| Filter SHA-256 | `{flt.get('sha256') or '—'}` |",
        f"| Filter review state | `{flt.get('native_review', 'pending')}` |",
        "",
        "The sign-off file repeats the corpus and filter hashes. If the corpus or gazetteer",
        "changes afterwards, the sign-off stops matching and the release check fails — that is",
        "the point: one sign-off covers one exact freeze.",
        "",
        "## How to review",
        "",
        "1. Read the gazetteer table below, then open the file it came from",
        "   (`eval_manifests/content_blocklist.<version>.json`) and add any missing terms.",
        "2. Read the sampled sentences. For anything that should not ship, note the id.",
        "3. Fill in the sign-off JSON (see the end of this packet) and hand it back. It contains",
        "   only ids, hashes and verdicts, so it is safe to commit.",
        "",
        "A `drop` verdict is a blocking finding: the run is rebuilt after the gazetteer or the",
        "sampling changes, so the published sample never contains the dropped sentence.",
        "",
    ]

    registers = [r for r in focus if r in grouped] + [r for r in grouped if r not in focus]
    if blocklist_terms:
        lines += ["## Gazetteer to review", ""]
        for category, terms in blocklist_terms.items():
            latin = list(terms.get("latin") or [])
            tamil = list(terms.get("tamil") or [])
            lines += [
                f"### {category} ({len(latin)} Latin, {len(tamil)} Tamil)",
                "",
                "- Latin: " + (", ".join(f"`{t}`" for t in latin) if latin else "—"),
                "- Tamil: " + (", ".join(f"`{t}`" for t in tamil) if tamil else "—"),
                "",
            ]

    for register in registers:
        recs = grouped[register]
        tag = "  ← review focus" if register in focus else ""
        lines += [
            f"## {register} sample ({len(recs):,} sentences){tag}",
            "",
            "| id | source | text |",
            "|---|---|---|",
        ]
        for rec in recs:
            text = str(rec["text"]).replace("|", "\\|")
            lines.append(f"| `{rec['id']}` | {rec.get('source', '')} | {text} |")
        lines.append("")

    template = signoff_template(records, corpus_sha256, filter_desc)
    lines += [
        "## Sign-off file",
        "",
        "Save this as `eval_manifests/k6_review_signoff.json`, fill in `reviewed_by`,",
        "`reviewed_at` and one verdict per sentence, then run:",
        "",
        "```bash",
        "python scripts/check_review_signoff.py",
        "```",
        "",
        "```json",
        json.dumps(template, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def signoff_template(
    records: Sequence[Mapping[str, Any]],
    corpus_sha256: str,
    filter_desc: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Skeleton sign-off: ids and hashes filled in, human fields empty."""
    flt = dict(filter_desc or {})
    return {
        "schema_version": SIGNOFF_SCHEMA_VERSION,
        "corpus_output_sha256": corpus_sha256,
        "filter_version": flt.get("version"),
        "filter_sha256": flt.get("sha256"),
        "reviewed_by": "",
        "reviewed_at": "",
        "native_speaker": False,
        "gazetteer": {
            "version": flt.get("version"),
            "verdict": VERDICT_PENDING,
            "missing_terms": [],
            "notes": "",
        },
        "sentences": {
            str(rec["id"]): {"sha256": text_sha256(str(rec["text"])), "verdict": VERDICT_PENDING}
            for rec in records
        },
    }


def validate_signoff(
    signoff: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    corpus_sha256: str,
    filter_desc: Optional[Mapping[str, Any]],
) -> List[str]:
    """Return every reason this sign-off does not cover the given freeze.

    An empty list means a human has signed off this exact corpus and gazetteer, and nothing
    in the sample was flagged for removal.
    """
    problems: List[str] = []
    flt = dict(filter_desc or {})

    if signoff.get("schema_version") != SIGNOFF_SCHEMA_VERSION:
        problems.append(
            f"schema_version {signoff.get('schema_version')} != {SIGNOFF_SCHEMA_VERSION}"
        )
    if signoff.get("corpus_output_sha256") != corpus_sha256:
        problems.append("corpus_output_sha256 does not match this freeze; the review is stale")
    if (signoff.get("filter_version") or "none") != (flt.get("version") or "none"):
        problems.append("filter_version does not match the gazetteer used for this freeze")
    if signoff.get("filter_sha256") != flt.get("sha256"):
        problems.append("filter_sha256 does not match the gazetteer file")

    if not str(signoff.get("reviewed_by") or "").strip():
        problems.append("reviewed_by is empty")
    # Strict identity: `"no"`, 1 and [True] are all truthy, and this field is an attestation.
    if signoff.get("native_speaker") is not True:
        problems.append("native_speaker must be the boolean true")
    reviewed_at = str(signoff.get("reviewed_at") or "").strip()
    if not reviewed_at:
        problems.append("reviewed_at is empty")
    else:
        try:
            reviewed_on = date.fromisoformat(reviewed_at)
        except ValueError:
            problems.append(f"reviewed_at {reviewed_at!r} is not an ISO date")
        else:
            if reviewed_on > date.today():
                problems.append(f"reviewed_at {reviewed_at} is in the future")

    # The gazetteer itself must have left review; a sign-off cannot approve an unreviewed list.
    blocklist_review = str(flt.get("native_review") or "pending").strip().lower()
    if blocklist_review == "pending":
        problems.append(
            "the gazetteer is still marked native_review='pending' in the blocklist file; "
            "a reviewer must update it and the freeze must be rebuilt"
        )

    gazetteer = signoff.get("gazetteer") or {}
    if gazetteer.get("verdict") != VERDICT_KEEP:
        problems.append(
            f"gazetteer verdict is {gazetteer.get('verdict')!r}; expected {VERDICT_KEEP!r}"
        )
    if gazetteer.get("missing_terms"):
        problems.append(
            "gazetteer review lists missing terms: "
            + ", ".join(map(str, gazetteer["missing_terms"]))
        )

    sentences = signoff.get("sentences") or {}
    expected = {str(rec["id"]): text_sha256(str(rec["text"])) for rec in records}
    missing = sorted(set(expected) - set(sentences))
    extra = sorted(set(sentences) - set(expected))
    if missing:
        problems.append(f"{len(missing)} sentence(s) not reviewed, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} reviewed id(s) are not in this freeze, e.g. {extra[:3]}")

    pending = dropped = bad_hash = bad_verdict = malformed = 0
    for sid, want_hash in expected.items():
        entry = sentences.get(sid)
        if not isinstance(entry, Mapping):
            # `"tanglish-00000": "keep"` used to be skipped here, bypassing every check.
            malformed += 1
            continue
        verdict = entry.get("verdict")
        if verdict == VERDICT_PENDING:
            pending += 1
        elif verdict == VERDICT_DROP:
            dropped += 1
        elif verdict not in VERDICTS:
            bad_verdict += 1
        if entry.get("sha256") != want_hash:
            bad_hash += 1
    if pending:
        problems.append(f"{pending} sentence(s) still marked {VERDICT_PENDING!r}")
    if dropped:
        problems.append(
            f"{dropped} sentence(s) marked {VERDICT_DROP!r}; rebuild the freeze without them"
        )
    if malformed:
        problems.append(
            f"{malformed} sentence entr(y/ies) are not objects with 'sha256' and 'verdict'"
        )
    if bad_verdict:
        problems.append(f"{bad_verdict} sentence(s) have an unknown verdict")
    if bad_hash:
        problems.append(f"{bad_hash} sentence hash(es) do not match this freeze")
    return problems


def actual_filter_description(spec: Mapping[str, Any], base: Path) -> Optional[Dict[str, Any]]:
    """Hash the blocklist that is *on disk now*, instead of trusting the manifest's own record.

    Without this, a gate can be satisfied by a manifest that describes a filter which no longer
    exists on disk — emptying the blocklist still passed, as the review found.
    """
    from adhan_slm.eval import content_filter as cf  # local import: keeps this module light

    name = str(spec.get("content_filter") or "none").strip()
    if name == "none":
        return None
    path = Path(name)
    path = path if path.is_absolute() else (Path(base) / path)
    loaded = cf.ContentFilter.load(path)
    desc = loaded.describe()
    try:
        desc["path"] = path.resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        pass
    return desc


def load_signoff(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
