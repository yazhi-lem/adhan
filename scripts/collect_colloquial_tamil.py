#!/usr/bin/env python3
"""
collect_colloquial_tamil.py — Tamil colloquial corpus collector
ARIVU + Hermes | Rotation 26 Cycle 2 | Jun 18, 2026

Collects spoken Tamil from:
- Podcast transcripts (Tamil podcasts)
- Movie dialogues (with licensing)
- YouTube video transcripts
- Everyday conversation datasets

Output: JSONL format matching OpenSangam schema
        {"text": "...", "source": "...", "type": "colloquial", "register": "spoken"}
"""

import argparse
import json
from pathlib import Path

# Sources to target (when network available)
COLONIAL_SOURCES = {
    "podcasts": [
        "https://anchor.fm/tamil-podcasts",
        "https://open.spotify.com/show/tamil",
        "https://podcasts.google.com/tamil",
    ],
    "movies": [
        # Need to check licensing: Sun TV, Raj TV archives
        "https://www.tamilrockers.com (LICENSE REQUIRED)",
    ],
    "youtube": [
        "https://youtube.com/tamil-content",
        "https://youtube.com/ted-tamil",
    ],
    "dialogue_sets": [
        # Academic sets if available
    ]
}

def collect_mock_data():
    """Generate mock colloquial data for testing (network blocked)."""
    mock_entries = [
        {"text": "ஏண்ணா, உங்களா என்ன பண்றது?", "source": "mock-dialogue", "type": "colloquial", "register": "spoken"},
        {"text": "நான் பாட்டு போட்டேன், வானடா!", "source": "mock-dialogue", "type": "colloquial", "register": "spoken"},
        {"text": "இது எப்படி இருக்கும் டா?", "source": "mock-dialogue", "type": "colloquial", "register": "spoken"},
        {"text": "ச்சோ, அப்படி இன்னு பேசாத்!", "source": "mock-dialogue", "type": "colloquial", "register": "spoken"},
        {"text": "பிள்ளைகளா, பள்ளிக்கு போகணும்!", "source": "mock-dialogue", "type": "colloquial", "register": "spoken"},
    ]
    return mock_entries

def main():
    parser = argparse.ArgumentParser(description='Collect Tamil colloquial data')
    parser.add_argument('--output', '-o', default='models/adhan/data/colloquial_tamil/mock.jsonl')
    parser.add_argument('--mock', action='store_true', help='Generate mock data (network blocked)')
    parser.add_argument('--sources', nargs='+', choices=list(COLONIAL_SOURCES.keys()), help='Specific sources')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TAMIL COLLOQUIAL CORPUS COLLECTOR")
    print("ARIVU + Hermes | Rotation 26 Cycle 2 | Jun 18, 2026")
    print("=" * 60)
    
    if args.mock:
        entries = collect_mock_data()
        print(f"\n[MOCK MODE] Generated {len(entries)} colloquial examples")
        print("Sources targeted when network available:")
        for src, urls in COLONIAL_SOURCES.items():
            print(f"  - {src}")
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"\nSaved to {output_path}")
    else:
        print("\n[REAL MODE] Network required for scraping")
        print("Sources:")
        for src, urls in COLONIAL_SOURCES.items():
            print(f"  - {src}: {len(urls)} sources")

if __name__ == '__main__':
    main()