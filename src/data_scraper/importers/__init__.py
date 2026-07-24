"""Yazhi ecosystem data importers for corpus curation.

Provides integration with yazhi projects:
- yazhi-lem/vazhi: Knowledge/QA dataset
- yazhi-lem/corpus-tamil: Pre-curated Tamil corpus
- Yazhi API + open Sangam: Classical literature
"""

from .corpus_tamil_importer import CorpusTamilImporter
from .sangam_importer import SangamImporter
from .vazhi_importer import VazhiImporter

__all__ = ["VazhiImporter", "CorpusTamilImporter", "SangamImporter"]
