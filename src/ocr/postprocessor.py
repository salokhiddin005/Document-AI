"""
OCR post-processor -- language-aware text correction.
Fixes systematic errors that PaddleOCR makes with specific languages.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


_UZBEK_RULES: List[Tuple[str, str]] = [
    # va misread as vn
    (r'\bvn\b',             'va'),
    (r'\bVn\b',             'Va'),

    # h -> n at word start
    (r'\bnisoblanadi\b',    'hisoblanadi'),
    (r'\bNisoblanadi\b',    'Hisoblanadi'),
    (r'\bnokimiyat',        'hokimiyat'),
    (r'\bNokimiyat',        'Hokimiyat'),
    (r'\bnukumat',          'hukumat'),
    (r'\bNukumat',          'Hukumat'),
    (r'\bnuquq',            'huquq'),
    (r'\bNuquq',            'Huquq'),

    # q -> g substitution
    (r'\bgonun',            'qonun'),
    (r'\bGonun',            'Qonun'),
    (r'\bgiladi\b',         'qiladi'),
    (r'\bgilgan\b',         'qilgan'),
    (r'\bgilish\b',         'qilish'),
    (r'\bgilinadi\b',       'qilinadi'),
    (r'\bgilin',            'qilin'),
    (r'\bchigaruvchi\b',    'chiqaruvchi'),
    (r'\bxalg\b',           'xalq'),
    (r'\bXalg\b',           'Xalq'),
    (r"\bgo'mondoni\b",     "qo'mondoni"),
    (r"\bGo'mondoni\b",     "Qo'mondoni"),
    (r'\bgabul\b',          'qabul'),
    (r'\bGabul\b',          'Qabul'),
    (r'\bgaysi\b',          'qaysi'),
    (r'\bgachon\b',         'qachon'),

    # Number/letter confusion
    (r'\bIoo\b',            '100'),
    (r'\bI00\b',            '100'),

    # Character swap errors
    (r'\bslyosiy\b',        'siyosiy'),
    (r'\bSlyosiy\b',        'Siyosiy'),
    (r'\bovnz\b',           'ovoz'),
    (r'\bvnz\b',            'ovoz'),

    # Missing spaces
    (r'(?<=[a-z])hududlaridan\b',  ' hududlaridan'),

    # p -> or
    (r'\borezident',        'prezident'),
    (r'\bOrezident',        'Prezident'),

    # k/ki -> d
    (r'\bdritadi\b',        'kiritadi'),
    (r'\bDritadi\b',        'Kiritadi'),

    # Missing first characters
    (r'\bshonchiga\b',      'ishonchiga'),
    (r'\bborat\b',          'iborat'),
    (r'\baoliyat\b',        'faoliyat'),
    (r'\bAoliyat\b',        'Faoliyat'),
]

_RULES_BY_LANG: Dict[str, List[Tuple[str, str]]] = {
    'uz':     _UZBEK_RULES,
    'uzbek':  _UZBEK_RULES,
}


def correct(text: str, lang: str = 'en') -> str:
    """Apply language-specific OCR corrections. Returns text unchanged if no rules."""
    rules = _RULES_BY_LANG.get(lang.lower(), [])
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text
