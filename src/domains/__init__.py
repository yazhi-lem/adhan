"""Yazhi domain layer — one reasoning kernel, personas on top.

``domains.core`` is the Core Domain reasoning kernel: an orchestration shell that
routes all five agent postures (context, reasoning, planning, reflection,
generation) through a *single* Adhan checkpoint.

Product-specific domains (Yazh, Annachi, the legal/gov verticals) are meant to
land next to ``core`` as **thin personas** — a system prompt plus a retrieval
scope — rather than as parallel reasoning stacks each reimplementing context and
reflection on their own.

See docs/CORE_DOMAIN_ADHAN_INTEGRATION.md.
"""
__version__ = "0.1.0.dev0"
