"""Tooth-numbering helpers (Universal vs FDI).

Two systems are common in dental records:
- Universal Numbering System (US, ADA): teeth are numbered 1–32 starting from
  the maxillary right third molar.
- FDI World Dental Federation: two-digit codes such as 18, 21, 36, 47.

This module converts between them and provides quadrant / arch helpers.
"""

from __future__ import annotations

UNIVERSAL_TO_FDI: dict[int, int] = {
    1: 18,  2: 17,  3: 16,  4: 15,  5: 14,  6: 13,  7: 12,  8: 11,
    9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    17: 38, 18: 37, 19: 36, 20: 35, 21: 34, 22: 33, 23: 32, 24: 31,
    25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47, 32: 48,
}

FDI_TO_UNIVERSAL: dict[int, int] = {v: k for k, v in UNIVERSAL_TO_FDI.items()}


def universal_to_fdi(n: int) -> int:
    if n not in UNIVERSAL_TO_FDI:
        raise ValueError(f"Universal tooth {n} out of range (1-32)")
    return UNIVERSAL_TO_FDI[n]


def fdi_to_universal(n: int) -> int:
    if n not in FDI_TO_UNIVERSAL:
        raise ValueError(f"FDI tooth {n} not valid (e.g. 18, 21, 47)")
    return FDI_TO_UNIVERSAL[n]


def quadrant_universal(n: int) -> str:
    """Quadrants in Universal: UR=1-8, UL=9-16, LL=17-24, LR=25-32."""
    if 1 <= n <= 8:
        return "UR"
    if 9 <= n <= 16:
        return "UL"
    if 17 <= n <= 24:
        return "LL"
    if 25 <= n <= 32:
        return "LR"
    raise ValueError(f"Universal tooth {n} out of range")


def arch_universal(n: int) -> str:
    if 1 <= n <= 16:
        return "maxillary"
    if 17 <= n <= 32:
        return "mandibular"
    raise ValueError(f"Universal tooth {n} out of range")


def is_molar(n: int) -> bool:
    """Universal numbering: 1-3, 14-16, 17-19, 30-32 are molars."""
    return n in {1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32}


def is_anterior(n: int) -> bool:
    """Anterior teeth (canines + incisors): 6-11, 22-27."""
    return n in {6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27}


__all__ = [
    "UNIVERSAL_TO_FDI",
    "FDI_TO_UNIVERSAL",
    "universal_to_fdi",
    "fdi_to_universal",
    "quadrant_universal",
    "arch_universal",
    "is_molar",
    "is_anterior",
]
