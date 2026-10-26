#!/usr/bin/env python3
"""Build SFT data with structured Tamil reasoning (சிந்தனை → ஆய்வு → பரிசீலனை → பதில்).

This script creates training examples where the model learns to think, rationalize,
and reflect entirely in modern Tamil before answering. The reasoning trace is
included in the target but masked during training — only the final answer gets
gradient signal.

Usage:
    # Enrich existing Sangam SFT with reasoning
    python scripts/build_reasoning_sft.py \
        --input data/final/sft_sangam/train.jsonl \
        --output data/final/sft_reasoning/train.jsonl \
        --mode enrich

    # Generate pure reasoning tasks (math, logic, sandhi)
    python scripts/build_reasoning_sft.py \
        --output data/final/sft_reasoning/train.jsonl \
        --mode generate --n-examples 5000

    # Combine both
    python scripts/build_reasoning_sft.py \
        --input data/final/sft_sangam/train.jsonl \
        --output data/final/sft_reasoning/train.jsonl \
        --mode both --n-examples 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm import protocol as p  # noqa: E402
from adhan_slm.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

random.seed(42)


# --- Reasoning templates in modern Tamil ------------------------------------

TAMIL_REASONING_TEMPLATES: Dict[str, List[str]] = {
    "tinai_analysis": [
        {
            "thought": "இந்தப் பாடலில் மலை, காடு, விலங்குகள் பற்றிய குறிப்புகள் உள்ளன. இது குறிஞ்சித் திணையைச் சேர்ந்ததாகத் தோன்றுகிறது.",
            "rationalize": "குறிஞ்சித் திணையில் காதலர் இரவில் இரண்டரை நாடி இடைவெளியில் சந்திப்பார்கள். பாடலில் 'மலை' என்ற சொல் இருப்பதால், இது குறிஞ்சி நிலத்தைக் குறிக்கிறது.",
            "reflect": "எனினும், பாடலில் வயலும், மழையும் குறிப்பிடப்பட்டுள்ளதா என்று சரிபார்க்க வேண்டும். இல்லையெனில் குறிஞ்சி உறுதி.",
        },
        {
            "thought": "இந்தப் பாடல் காதலைப் பற்றியது. அகத்திணையைச் சேர்ந்தது.",
            "rationalize": "பாடலில் தலைவன்-தலைவி உறவு, பரத்தமையர், தோழி போன்ற அகவய உறுப்புகள் உள்ளன. புறத்திணையில் அரசர், போர், புலவர் போன்றவை இருக்கும்.",
            "reflect": "ஆனால் சில புறப்பாடல்களிலும் காதல் குறிப்புகள் இருக்கலாம். எனவே முழு சூழலையும் கவனிக்க வேண்டும்.",
        },
    ],
    "sandhi_reasoning": [
        {
            "thought": "இது இரண்டு சொற்களின் சந்திப் புணர்ச்சி. முதல் சொலின் இறுதி எழுத்தும், இரண்டாம் சொலின் முதல் எழுத்தும் மாற வேண்டும்.",
            "rationalize": "'கடல்' என்பது மெய்யிறுதி சொல். 'கரை' என்பது 'க' முதல் எழுத்தில் தொடங்குகிறது. மெய்யிறுதி + 'க' → 'ற்' + 'க' = 'ற்க'.",
            "reflect": "சந்தி விதி: மெய் + மெய் = இடையில் இடைநிலை எழுத்து சேரும். 'ல்' → 'ற்' மாற்றம் சரியா? ஆம், இது தோற்றச்சந்தி.",
        },
        {
            "thought": "இது உயிர் ஈற்றுச் சொல் + மெய் முதல் சொல்.",
            "rationalize": "'மரம்' இறுதி 'ம' மெய். 'அறை' முதல் 'அ' உயிர். மெய் + உயிர் = மெய்யோடு உயிர் கூடி உயிர்மெய் ஆகும். 'ம' + 'அ' = 'ம'. எனவே 'மரமறை'.",
            "reflect": "எனினும் சில சமயங்களில் 'ஆ' மாற்றம் வரும். இங்கு 'அ' என்பதால் மாற்றம் இல்லை. சரி.",
        },
    ],
    "math_reasoning": [
        {
            "thought": "இது ஒரு கூட்டல் கணக்கு. இரண்டு எண்களைக் கூட்ட வேண்டும்.",
            "rationalize": "12-ஐ 25-உடன் கூட்டும்போது, ஒவ்வொரு இடத்திலும் கூட்டுகிறோம். ஒன்றுகள் இடம்: 2 + 5 = 7. பத்துகள் இடம்: 1 + 2 = 3. எனவே 37.",
            "reflect": "எளிய கூட்டல். கடன் வரவில்லை. 37 சரியான விடை.",
        },
        {
            "thought": "இது விகிதாசாரக் கணக்கு. மூன்று எண்கள் தரப்பட்டுள்ளன, நான்காவது காண வேண்டும்.",
            "rationalize": "2:4 = x:8. இதில் 2 → 4 இரட்டிப்பு. எனவே x → 8 இரட்டிப்பு என்றால் x = 4. அல்லது குறுக்குப் பெருக்கல்: 2×8 = 4×x → x = 16/4 = 4.",
            "reflect": "இரண்டு முறையிலும் 4 வருகிறது. விடை உறுதி.",
        },
    ],
    "history_reasoning": [
        {
            "thought": "இது தமிழ்நாட்டு வரலாற்று கேள்வி. காலவரிசையைப் பார்க்க வேண்டும்.",
            "rationalize": "சங்க காலம் கி.மு. 300 முதல் கி.பி. 300 வரை. பல்லவர்கள் கி.பி. 275 முதல் 897 வரை. எனவே பல்லவர்கள் சங்க காலத்திற்குப் பிறகு வந்தவர்கள்.",
            "reflect": "சங்க காலத்திற்கு முன் வேறு அரச குலங்கள் இருந்தன. பல்லவர்கள் தெற்கில் ஆதிக்கம் செலுத்தினர். இது சரியான வரிசை.",
        },
    ],
    "refusal_reasoning": [
        {
            "thought": "இந்தக் கேள்வி தீங்கு விளைவிக்கக்கூடியது. பாதுகாப்பு விதிகளைப் பயன்படுத்த வேண்டும்.",
            "rationalize": "நான் ஒரு AI. மனிதர்களுக்கு தீங்கு விளைவிக்கும் தகவல்களை வழங்கக்கூடாது. இந்தக் கேள்வி அப்படிப்பட்டது.",
            "reflect": "இதை மறுக்க வேண்டும். ஆனால் அதே சமயம் கடினமாக இல்லாமல், கல்வி நோக்கில் வேறு எப்படி உதவலாம் என்றும் கூறலாம்.",
        },
    ],
    "grammar_reasoning": [
        {
            "thought": "இது வேற்றுமை பற்றிய கேள்வி. எந்த வேற்றுமை உருபு பயன்பட்டுள்ளது என்று காண வேண்டும்.",
            "rationalize": "'மரத்தில்' — 'மரம்' + 'இல்'. 'இல்' என்பது ஏழாம் வேற்றுமை (இடம்/அதிகாரம்) உருபு. மரம் என்ற சொல்லின் இறுதி 'ம' மெய். 'ம' + 'இ' = 'தி'. எனவே 'மரத்தில்'.",
            "reflect": "ஏழாம் வேற்றுமை இடத்தைக் குறிக்கும். 'மரத்தின் மீது' என்றால் இரண்டாம் வேற்றுமை. இங்கு 'இல்' சரியான உருபு.",
        },
    ],
}


# --- Synthetic task generators ----------------------------------------------


def _rand_sandhi_task() -> Tuple[str, str, dict]:
    """Generate a sandhi reasoning example."""
    pairs = [
        ("கடல்", "கரை", "கடற்கரை", "மெய்யிறுதி + 'க' → 'ற்' + 'க' = 'ற்க'"),
        ("மரம்", "அறை", "மரமறை", "மெய் + உயிர் = உயிர்மெய், இங்கு மாற்றம் இல்லை"),
        ("நெல்", "கண்டி", "நெற்கண்டி", "'ல்' → 'ற்' மாற்றம் (தோற்றச்சந்தி)"),
        ("புல்", "காடு", "புல்காடு", "'ல்' → 'ல்' மாற்றம் இல்லை, நேர்ச்சந்தி"),
        ("கண்", "கொடி", "கண்கொடி", "'ண்' + 'க' = 'ண்' + 'க', நேர்ச்சந்தி"),
        ("வான்", "கோள்", "வான்கோள்", "'ன்' + 'க' = 'ங்' + 'க' = 'ங்க' — இங்கு 'ன்' + 'க' = 'ன்க'"),
    ]
    a, b, ans, rule = random.choice(pairs)
    prompt = f"பயனர்: '{a}' + '{b}' சந்திப்பது எப்படி?\nஉதவியாளர்:"
    thought = f"இது இரண்டு சொற்களின் சந்திப் புணர்ச்சி. {a} இறுதி எழுத்தும், {b} முதல் எழுத்தும் எப்படி மாறுகிறது என்று பார்க்க வேண்டும்."
    rationalize = f"{rule}. எனவே {a} + {b} = {ans}."
    reflect = "சந்தி விதிகளின்படி இது சரியானதா என்று மீண்டும் சரிபார்க்கிறேன். ஆம், சரி."
    return prompt, ans, {"thought": thought, "rationalize": rationalize, "reflect": reflect}


def _rand_math_task() -> Tuple[str, str, dict]:
    """Generate a math reasoning example."""
    ops = [
        (lambda a, b: a + b, "கூட்டல்", "+"),
        (lambda a, b: a - b, "கழித்தல்", "-"),
        (lambda a, b: a * b, "பெருக்கல்", "×"),
    ]
    fn, name, sym = random.choice(ops)
    a, b = random.randint(2, 50), random.randint(2, 50)
    if name == "கழித்தல்" and a < b:
        a, b = b, a
    ans = fn(a, b)
    prompt = f"பயனர்: {a} {sym} {b} = ?\nஉதவியாளர்:"
    thought = f"இது ஒரு {name} கணக்கு. {a}-ஐயும் {b}-ஐயும் {name} செய்ய வேண்டும்."
    if name == "கூட்டல்":
        rationalize = f"{a} + {b}: ஒவ்வொரு இடத்திலும் கூட்டுகிறோம். விடை {ans}."
    elif name == "கழித்தல்":
        rationalize = f"{a} - {b}: {b}-ஐ {a}-லிருந்து கழிக்கிறோம். விடை {ans}."
    else:
        rationalize = f"{a} × {b}: {a}-ஐ {b} முறை கூட்டுகிறோம். விடை {ans}."
    reflect = f"மீண்டும் சரிபார்க்கிறேன்: {ans} சரியான விடை."
    return prompt, str(ans), {"thought": thought, "rationalize": rationalize, "reflect": reflect}


def _rand_tinai_task() -> Tuple[str, str, dict]:
    """Generate a tinai classification reasoning example."""
    tinais = [
        ("குறிஞ்சி", "மலை, காடு, காதல், இரவு, குறிஞ்சிப்பூ"),
        ("முல்லை", "காடு, மேய்தல், பரத்தமை, பகல், முல்லைப்பூ"),
        ("மருதம்", "வயல், விளைச்சல், இரவு, இரவலர், மருதம்"),
        ("நெய்தல்", "கடல், காதல், பகல், நெய்தல்பூ"),
        ("பாலை", "வறண்ட நிலம், பிரிவு, பகல், பாலை"),
    ]
    tinai, clues = random.choice(tinais)
    prompt = (
        f"பயனர்: ஒரு சங்கப் பாடலில் {clues} பற்றிய குறிப்புகள் உள்ளன. "
        f"இந்தப் பாடலின் திணை எது?\nஉதவியாளர்:"
    )
    thought = f"இந்தப் பாடலின் சூழலைப் பார்க்கும்போது {clues} குறிப்பிடப்பட்டுள்ளன."
    rationalize = (
        f"{tinai} திணையில் {clues} முக்கிய அம்சங்கள். எனவே இந்தப் பாடல் {tinai} திணையைச் சேர்ந்தது."
    )
    reflect = f"வேறு திணைகளில் இந்த அம்சங்கள் இருக்கின்றனவா என்று சரிபார்க்கிறேன். இல்லை, {tinai} தான் சரியானது."
    return prompt, tinai, {"thought": thought, "rationalize": rationalize, "reflect": reflect}


def _rand_refusal_task() -> Tuple[str, str, dict]:
    """Generate a refusal with reasoning."""
    harmful = [
        "ஒருவரை எப்படிக் காயப்படுத்துவது?",
        "தீங்கு விளைவிக்கும் மருந்து தயாரிப்பது எப்படி?",
        "பொய்யான தகவலைப் பரப்புவது எப்படி?",
    ]
    q = random.choice(harmful)
    prompt = f"பயனர்: {q}\nஉதவியாளர்:"
    thought = "இந்தக் கேள்வி தீங்கு விளைவிக்கக்கூடியது. பாதுகாப்பு விதிகளைப் பயன்படுத்த வேண்டும்."
    rationalize = "நான் ஒரு AI. மனிதர்களுக்கு தீங்கு விளைவிக்கும் தகவல்களை வழங்கக்கூடாது."
    reflect = "இதை மறுக்க வேண்டும். ஆனால் கடினமாக இல்லாமல், கல்வி நோக்கில் வேறு வழிகளில் உதவலாம்."
    return prompt, p.REFUSAL, {"thought": thought, "rationalize": rationalize, "reflect": reflect}


# --- Enrichment: add reasoning to existing SFT data -------------------------


def enrich_with_reasoning(input_path: str, output_path: str) -> None:
    """Read existing SFT JSONL and add reasoning traces to each example."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = rec.get("task", "")
            answer = rec.get("answer", "")
            # Pick reasoning template based on task
            templates = []
            if "tinai" in task:
                templates = TAMIL_REASONING_TEMPLATES.get("tinai_analysis", [])
            elif "sandhi" in task or "grammar" in task:
                templates = TAMIL_REASONING_TEMPLATES.get("grammar_reasoning", [])
            elif "qa" in task:
                templates = TAMIL_REASONING_TEMPLATES.get("history_reasoning", [])
            elif "refusal" in task or not rec.get("answerable", True):
                templates = TAMIL_REASONING_TEMPLATES.get("refusal_reasoning", [])
            else:
                templates = TAMIL_REASONING_TEMPLATES.get("history_reasoning", [])

            if templates:
                tmpl = random.choice(templates)
                rec["target"] = p.render_target(
                    answer=answer,
                    thought=tmpl.get("thought", ""),
                    rationalize=tmpl.get("rationalize", ""),
                    reflect=tmpl.get("reflect", ""),
                    citations=rec.get("citations", []),
                )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    logger.info(f"enriched {n} examples -> {out_path}")


# --- Generation: create pure reasoning tasks -------------------------------


def generate_reasoning_tasks(n_examples: int, out_path: Path) -> None:
    """Generate synthetic reasoning tasks from scratch."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generators = [
        _rand_sandhi_task,
        _rand_math_task,
        _rand_tinai_task,
        _rand_refusal_task,
    ]
    weights = [0.25, 0.25, 0.25, 0.25]

    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(n_examples):
            gen = random.choices(generators, weights=weights)[0]
            prompt_text, answer, reasoning = gen()
            # Build full prompt with system + user
            system = (
                "நீ ஒரு தமிழ் உதவியாளர். ஒவ்வொரு கேள்விக்கும் சிந்தித்து, "
                "ஆய்வு செய்து, பரிசீலனை செய்து பின் பதில் சொல்."
            )
            full_prompt = f"{p.SYSTEM} {system}\n{prompt_text}"
            target = p.render_target(
                answer=answer,
                thought=reasoning["thought"],
                rationalize=reasoning["rationalize"],
                reflect=reasoning["reflect"],
            )
            rec = {
                "id": f"reasoning_synth_{i}",
                "task": "reasoning_synthetic",
                "prompt": full_prompt,
                "target": target,
                "answer": answer,
                "messages": [{"role": "user", "content": prompt_text}],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 1000 == 0:
                logger.info(f"generated {i + 1}/{n_examples} reasoning tasks")
    logger.info(f"generated {n_examples} synthetic reasoning tasks -> {out_path}")


# --- Main -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reasoning-heavy Tamil SFT data")
    ap.add_argument("--input", help="input JSONL (for enrich/both modes)")
    ap.add_argument("--output", required=True, help="output JSONL")
    ap.add_argument("--mode", choices=["enrich", "generate", "both"], default="generate")
    ap.add_argument("--n-examples", type=int, default=5000, help="for generate/both")
    args = ap.parse_args()

    if args.mode == "enrich":
        if not args.input:
            raise SystemExit("--input required for enrich mode")
        enrich_with_reasoning(args.input, args.output)
    elif args.mode == "generate":
        generate_reasoning_tasks(args.n_examples, Path(args.output))
    elif args.mode == "both":
        if not args.input:
            raise SystemExit("--input required for both mode")
        # Enrich existing + generate synthetic
        enrich_with_reasoning(args.input, args.output)
        # Append synthetic
        tmp = Path(args.output).with_suffix(".tmp.jsonl")
        generate_reasoning_tasks(args.n_examples, tmp)
        with open(args.output, "a", encoding="utf-8") as fout:
            with open(tmp, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
        tmp.unlink()
        logger.info(f"combined output -> {args.output}")


if __name__ == "__main__":
    main()
