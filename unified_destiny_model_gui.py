
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Destiny Harmonic Model GUI
---------------------------------
A mathematically explicit, engineering-oriented experiment that fuses:
- Four Pillars / BaZi (sexagenary year-month-day-hour)
- Yi Jing / hexagram structure
- Liu Yao style moving-line transformation
- Western date/name numerology

This is NOT presented as scientifically proven future prediction.
What the program does provide is:
1. a single deterministic mathematical model,
2. internal structural benchmarks,
3. CSV backtesting hooks against external labelled outcomes.

Requirements:
- Python 3.11+
- swisseph
- numpy
- tkinter
- matplotlib (optional for chart)
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import swisseph as swe
from zoneinfo import ZoneInfo

# -----------------------------
# Constants
# -----------------------------

STEMS = list("甲乙丙丁戊己庚辛壬癸")
BRANCHES = list("子丑寅卯辰巳午未申酉戌亥")
ELEMENTS = ["木", "火", "土", "金", "水"]
WEEKDAYS_ZH = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]

STEM_ELEMENT = {
    0: 0, 1: 0,  # 甲乙 木
    2: 1, 3: 1,  # 丙丁 火
    4: 2, 5: 2,  # 戊己 土
    6: 3, 7: 3,  # 庚辛 金
    8: 4, 9: 4,  # 壬癸 水
}

# Branch hidden stems with approximate primary/secondary/tertiary weights.
# This is a quantitative implementation choice for engineering use.
HIDDEN_STEMS = {
    0: [(9, 1.0)],                        # 子: 癸
    1: [(5, 0.60), (9, 0.25), (7, 0.15)],# 丑: 己 癸 辛
    2: [(0, 0.60), (2, 0.25), (4, 0.15)],# 寅: 甲 丙 戊
    3: [(1, 1.0)],                        # 卯: 乙
    4: [(4, 0.60), (1, 0.25), (9, 0.15)],# 辰: 戊 乙 癸
    5: [(2, 0.60), (6, 0.25), (4, 0.15)],# 巳: 丙 庚 戊
    6: [(3, 0.70), (5, 0.30)],           # 午: 丁 己
    7: [(5, 0.60), (3, 0.25), (1, 0.15)],# 未: 己 丁 乙
    8: [(6, 0.60), (8, 0.25), (4, 0.15)],# 申: 庚 壬 戊
    9: [(7, 1.0)],                        # 酉: 辛
    10: [(4, 0.60), (7, 0.25), (3, 0.15)],# 戌: 戊 辛 丁
    11: [(8, 0.70), (0, 0.30)],          # 亥: 壬 甲
}

TRIGRAM_BITS = {
    "乾": "111",
    "兌": "110",
    "離": "101",
    "震": "100",
    "巽": "011",
    "坎": "010",
    "艮": "001",
    "坤": "000",
}
TRIGRAM_NAME_BY_BITS = {v: k for k, v in TRIGRAM_BITS.items()}
TRIGRAM_ELEMENT = {"乾": 3, "兌": 3, "離": 1, "震": 0, "巽": 0, "坎": 4, "艮": 2, "坤": 2}

# Parent/child/control relations on [木, 火, 土, 金, 水]
PARENT = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}
CHILD = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
CONTROLLER = {0: 3, 1: 4, 2: 0, 3: 1, 4: 2}
CONTROLLED = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}

# 12 solar-term starts used for BaZi solar months.
JIE_TERMS = [
    ("立春", 315.0, 0),
    ("驚蟄", 345.0, 1),
    ("清明", 15.0, 2),
    ("立夏", 45.0, 3),
    ("芒種", 75.0, 4),
    ("小暑", 105.0, 5),
    ("立秋", 135.0, 6),
    ("白露", 165.0, 7),
    ("寒露", 195.0, 8),
    ("立冬", 225.0, 9),
    ("大雪", 255.0, 10),
    ("小寒", 285.0, 11),
]

# Six line projection matrix from 5-element space into 6 line potentials.
Q = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 0, -1, -1],
        [1, 0, -1, 0, 1],
        [0, 1, -1, 1, -1],
        [1, -1, 1, -1, 0],
        [1, -1, -1, 1, 0],
    ],
    dtype=float,
)
Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

ASTRO_FLAGS = swe.FLG_MOSEPH  # portable mode; no external ephemeris files required

CHANNEL_LABELS = {
    "resource": "資源",
    "output": "輸出",
    "wealth": "財務",
    "authority": "規範",
    "companion": "同頻",
}

MODEL_LABELS = {
    "bazi": "八字基線",
    "hex": "卦象基線",
    "num": "數字基線",
    "naive": "三系統平均",
    "unified": "統一諧振模型",
}

# -----------------------------
# Data loading
# -----------------------------

def _default_hex_data_path() -> Path:
    return Path(__file__).resolve().with_name("hexagrams_compact.json")


def load_hexagrams(path: Optional[Path] = None) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    path = Path(path) if path else _default_hex_data_path()
    if not path.exists():
        raise FileNotFoundError(f"找不到卦資料檔: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    by_bits = {v["binary_bottom_to_top"]: v for v in data.values()}
    return data, by_bits


HEXAGRAMS, HEX_BY_BITS = load_hexagrams()

SEXAGENARY = [(i % 10, i % 12) for i in range(60)]
PAIR_TO_INDEX60 = {(s, b): i for i, (s, b) in enumerate(SEXAGENARY)}


# -----------------------------
# Helpers
# -----------------------------

def sexagenary_name(index: int) -> str:
    index %= 60
    return STEMS[index % 10] + BRANCHES[index % 12]


def sexagenary_index(stem: int, branch: int) -> int:
    return PAIR_TO_INDEX60[(stem, branch)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def normalize_vec(v: Iterable[float]) -> np.ndarray:
    arr = np.array(list(v), dtype=float)
    s = float(arr.sum())
    if s <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / s


def phase(z: complex) -> float:
    if abs(z) < 1e-12:
        return 0.0
    return math.atan2(z.imag, z.real)


def complex_alignment(z1: complex, z2: complex) -> float:
    n1, n2 = abs(z1), abs(z2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float((z1.conjugate() * z2).real / (n1 * n2))


def complex_cycle(index: int, modulus: int, harmonics: Tuple[int, ...] = (1, 2, 3)) -> complex:
    z = 0j
    for r in harmonics:
        angle = 2.0 * math.pi * r * index / modulus
        z += (1.0 / r) * complex(math.cos(angle), math.sin(angle))
    return z


def reduce_num(n: int, keep_master: bool = True) -> int:
    n = abs(int(n))
    while n > 9 and not (keep_master and n in (11, 22, 33)):
        n = sum(int(d) for d in str(n))
    return n


def pythagorean_name_number(name: str) -> Optional[int]:
    mapping = {
        **{c: 1 for c in "AJS"},
        **{c: 2 for c in "BKT"},
        **{c: 3 for c in "CLU"},
        **{c: 4 for c in "DMV"},
        **{c: 5 for c in "ENW"},
        **{c: 6 for c in "FOX"},
        **{c: 7 for c in "GPY"},
        **{c: 8 for c in "HQZ"},
        **{c: 9 for c in "IR"},
    }
    total = 0
    found = False
    for ch in name.upper():
        if ch in mapping:
            total += mapping[ch]
            found = True
    return reduce_num(total) if found else None


def rankdata(values: List[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return np.array(ranks, dtype=float)


def pearson_corr(x: List[float], y: List[float]) -> float:
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if len(x_arr) < 2 or float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_corr(x: List[float], y: List[float]) -> float:
    return pearson_corr(rankdata(x).tolist(), rankdata(y).tolist())


def directional_accuracy(x: List[float], y: List[float]) -> float:
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if len(x_arr) < 2:
        return float("nan")
    dx = np.sign(np.diff(x_arr))
    dy = np.sign(np.diff(y_arr))
    mask = (dx != 0) & (dy != 0)
    if int(mask.sum()) == 0:
        return float("nan")
    return float((dx[mask] == dy[mask]).mean())


def trigram_from_bits(bits3_bottom_to_top: str) -> str:
    return TRIGRAM_NAME_BY_BITS[bits3_bottom_to_top]


def element_relation_score(a: int, b: int) -> float:
    if a == b:
        return 1.0
    if CHILD[a] == b or CHILD[b] == a:
        return 0.45
    if CONTROLLED[a] == b or CONTROLLED[b] == a:
        return -0.45
    return 0.0


def bits_to_hex(bits_bottom_to_top: List[int]) -> dict:
    key = "".join("1" if int(x) else "0" for x in bits_bottom_to_top)
    return HEX_BY_BITS[key]


def score_label(score: float, volatility: float) -> str:
    if score >= 75:
        base = "推進"
    elif score >= 60:
        base = "順勢"
    elif score >= 45:
        base = "平衡"
    elif score >= 30:
        base = "收斂"
    else:
        base = "避險"
    if volatility >= 0.65:
        return "高波動" + base
    return base


def parse_datetime_guess(text: str, tz_name: str = "Asia/Taipei") -> datetime:
    text = text.strip()
    try:
        if "T" in text or " " in text:
            dt = datetime.fromisoformat(text)
        else:
            d = date.fromisoformat(text)
            dt = datetime(d.year, d.month, d.day, 12, 0)
    except Exception:
        for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(text, fmt)
                if fmt in ("%Y/%m/%d", "%Y-%m-%d"):
                    dt = dt.replace(hour=12, minute=0)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"無法解析日期或時間: {text}")

    tz = ZoneInfo(tz_name)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def auto_detect_columns(fieldnames: List[str]) -> Tuple[str, str]:
    lowered = [f.lower() for f in fieldnames]
    date_candidates = ["datetime", "timestamp", "date", "time", "day"]
    outcome_candidates = ["outcome", "score", "result", "label", "target", "value", "actual"]
    date_col = None
    outcome_col = None

    for candidate in date_candidates:
        for original, low in zip(fieldnames, lowered):
            if candidate == low or candidate in low:
                date_col = original
                break
        if date_col:
            break

    for candidate in outcome_candidates:
        for original, low in zip(fieldnames, lowered):
            if candidate == low or candidate in low:
                outcome_col = original
                break
        if outcome_col:
            break

    if date_col is None:
        date_col = fieldnames[0]
    if outcome_col is None:
        if len(fieldnames) < 2:
            raise ValueError("CSV 至少需要兩欄")
        outcome_col = fieldnames[1]
    return date_col, outcome_col


# -----------------------------
# Astronomy
# -----------------------------

def dt_to_jdut(dt: datetime) -> float:
    if dt.tzinfo is None:
        raise ValueError("datetime 必須帶時區")
    utc = dt.astimezone(timezone.utc)
    _jdet, jdut = swe.utc_to_jd(
        utc.year,
        utc.month,
        utc.day,
        utc.hour,
        utc.minute,
        utc.second + utc.microsecond / 1e6,
        swe.GREG_CAL,
    )
    return jdut


def jdut_to_dt(jd: float, tz: timezone | ZoneInfo = timezone.utc) -> datetime:
    y, m, d, hh, mm, sec = swe.jdut1_to_utc(jd, swe.GREG_CAL)
    sec_int = int(sec)
    micro = int(round((sec - sec_int) * 1e6))
    if micro == 1_000_000:
        sec_int += 1
        micro = 0
    dt = datetime(y, m, d, hh, mm, sec_int, micro, tzinfo=timezone.utc)
    return dt.astimezone(tz)


@lru_cache(maxsize=50000)
def sun_longitude(jd: float) -> float:
    return float(swe.calc_ut(jd, swe.SUN, ASTRO_FLAGS)[0][0] % 360.0)


@lru_cache(maxsize=50000)
def moon_longitude(jd: float) -> float:
    return float(swe.calc_ut(jd, swe.MOON, ASTRO_FLAGS)[0][0] % 360.0)


def lunar_phase_angle(jd: float) -> float:
    return float((moon_longitude(jd) - sun_longitude(jd)) % 360.0)


def true_solar_datetime(local_dt: datetime, longitude_deg: Optional[float]) -> datetime:
    if longitude_deg is None:
        return local_dt
    jd = dt_to_jdut(local_dt)
    equation_of_time_days = float(swe.time_equ(jd))
    offset_hours = local_dt.utcoffset().total_seconds() / 3600.0
    standard_meridian = offset_hours * 15.0
    longitude_correction_hours = (longitude_deg - standard_meridian) / 15.0
    return (
        local_dt
        + timedelta(hours=longitude_correction_hours)
        + timedelta(days=equation_of_time_days)
    )


@lru_cache(maxsize=64)
def lichun_dt(cycle_year: int, tz_name: str) -> datetime:
    tz = ZoneInfo(tz_name)
    start_jd = swe.julday(cycle_year, 1, 20, 0.0, swe.GREG_CAL)
    cross = swe.solcross_ut(315.0, start_jd, ASTRO_FLAGS)
    return jdut_to_dt(cross, tz)


@lru_cache(maxsize=64)
def solar_month_boundaries(cycle_year: int, tz_name: str) -> Tuple[Tuple[str, datetime, int], ...]:
    tz = ZoneInfo(tz_name)
    approx = {
        "立春": (cycle_year, 2, 1),
        "驚蟄": (cycle_year, 3, 1),
        "清明": (cycle_year, 4, 1),
        "立夏": (cycle_year, 5, 1),
        "芒種": (cycle_year, 6, 1),
        "小暑": (cycle_year, 7, 1),
        "立秋": (cycle_year, 8, 1),
        "白露": (cycle_year, 9, 1),
        "寒露": (cycle_year, 10, 1),
        "立冬": (cycle_year, 11, 1),
        "大雪": (cycle_year, 12, 1),
        "小寒": (cycle_year + 1, 1, 1),
    }
    rows = []
    for name, angle, month_idx in JIE_TERMS:
        y, m, d = approx[name]
        jd0 = swe.julday(y, m, d, 0.0, swe.GREG_CAL)
        cross = swe.solcross_ut(angle, jd0, ASTRO_FLAGS)
        rows.append((name, jdut_to_dt(cross, tz), month_idx))
    rows.sort(key=lambda row: row[1])
    return tuple(rows)


# -----------------------------
# Model state
# -----------------------------

@dataclass(frozen=True)
class Pillar:
    stem: int
    branch: int
    index60: int
    name: str


def pillar_from_index(index: int) -> Pillar:
    index %= 60
    return Pillar(index % 10, index % 12, index, sexagenary_name(index))


def pillars_for_datetime(
    local_dt: datetime,
    tz_name: str = "Asia/Taipei",
    longitude_deg: Optional[float] = None,
    use_true_solar: bool = True,
    zi_rollover: bool = True,
) -> Dict[str, object]:
    tz = ZoneInfo(tz_name)
    if local_dt.tzinfo is None:
        local_dt = local_dt.replace(tzinfo=tz)
    else:
        local_dt = local_dt.astimezone(tz)

    calc_dt = true_solar_datetime(local_dt, longitude_deg) if use_true_solar else local_dt

    li_this = lichun_dt(calc_dt.year, tz_name)
    cycle_year = calc_dt.year if calc_dt >= li_this else calc_dt.year - 1

    year_index = (cycle_year - 1984) % 60  # 1984 立春後為甲子年
    year_p = pillar_from_index(year_index)

    boundaries = solar_month_boundaries(cycle_year, tz_name)
    month_idx = 11
    for _name, boundary_dt, idx in boundaries:
        if calc_dt >= boundary_dt:
            month_idx = idx
        else:
            break

    month_branch = (2 + month_idx) % 12  # 寅月起
    month_stem = (((year_p.stem % 5) * 2) + 2 + month_idx) % 10
    month_index60 = sexagenary_index(month_stem, month_branch)
    month_p = Pillar(month_stem, month_branch, month_index60, STEMS[month_stem] + BRANCHES[month_branch])

    day_dt = calc_dt
    if zi_rollover and day_dt.hour >= 23:
        day_dt = day_dt + timedelta(days=1)

    ref = date(1912, 2, 18)  # 甲子日
    day_index = (day_dt.date() - ref).days % 60
    day_p = pillar_from_index(day_index)

    hour_branch = ((calc_dt.hour + 1) // 2) % 12
    hour_stem = (((day_p.stem % 5) * 2) + hour_branch) % 10
    hour_index60 = sexagenary_index(hour_stem, hour_branch)
    hour_p = Pillar(hour_stem, hour_branch, hour_index60, STEMS[hour_stem] + BRANCHES[hour_branch])

    return {
        "year": year_p,
        "month": month_p,
        "day": day_p,
        "hour": hour_p,
        "calc_dt": calc_dt,
        "cycle_year": cycle_year,
        "month_index": month_idx,
    }


def pillar_tuple_list(pillars: Dict[str, object]) -> List[Tuple[int, int]]:
    return [
        (pillars["year"].stem, pillars["year"].branch),
        (pillars["month"].stem, pillars["month"].branch),
        (pillars["day"].stem, pillars["day"].branch),
        (pillars["hour"].stem, pillars["hour"].branch),
    ]


def element_vector_from_pillars(pillars: List[Tuple[int, int]]) -> np.ndarray:
    vec = np.zeros(5, dtype=float)
    for pos, (stem, branch) in enumerate(pillars):
        stem_weight = 1.25 if pos == 2 else (1.15 if pos == 1 else 1.0)
        branch_weight = 1.10 if pos == 1 else 0.95
        vec[STEM_ELEMENT[stem]] += stem_weight
        for hidden_stem, weight in HIDDEN_STEMS[branch]:
            vec[STEM_ELEMENT[hidden_stem]] += branch_weight * weight
    return vec


def numerology_profile(
    birth_dt: datetime,
    forecast_dt: Optional[datetime] = None,
    name: str = "",
) -> Dict[str, Optional[int]]:
    month = birth_dt.month
    day = birth_dt.day
    year = birth_dt.year
    life_path = reduce_num(reduce_num(month) + reduce_num(day) + reduce_num(year))
    attitude = reduce_num(reduce_num(month) + reduce_num(day))
    birthday = reduce_num(day)
    expression = pythagorean_name_number(name) if name else None

    out = {
        "life_path": life_path,
        "attitude": attitude,
        "birthday": birthday,
        "expression": expression,
    }

    if forecast_dt is not None:
        personal_year = reduce_num(reduce_num(month) + reduce_num(day) + reduce_num(forecast_dt.year))
        personal_month = reduce_num(personal_year + forecast_dt.month)
        personal_day = reduce_num(personal_month + forecast_dt.day)
        universal_day = reduce_num(
            reduce_num(forecast_dt.year) + reduce_num(forecast_dt.month) + reduce_num(forecast_dt.day)
        )
        out.update(
            {
                "personal_year": personal_year,
                "personal_month": personal_month,
                "personal_day": personal_day,
                "universal_day": universal_day,
            }
        )
    return out


def numerology_complex(numbers: Dict[str, Optional[int]]) -> complex:
    fields = [numbers.get("life_path"), numbers.get("attitude"), numbers.get("birthday")]
    weights = [1.3, 1.0, 0.9]
    if numbers.get("expression") is not None:
        fields.append(numbers["expression"])
        weights.append(0.8)

    z = 0j
    for value, weight in zip(fields, weights):
        if value is None:
            continue
        amp = 1.0 + (0.15 if value in (11, 22, 33) else 0.0)
        base = reduce_num(value, keep_master=False)
        for harmonic in (1, 2):
            angle = 2.0 * math.pi * harmonic * base / 9.0
            z += amp * weight * (1.0 / harmonic) * complex(math.cos(angle), math.sin(angle))
    return z


def ganzhi_complex(pillars: Dict[str, object]) -> complex:
    indices = [
        pillars["year"].index60,
        pillars["month"].index60,
        pillars["day"].index60,
        pillars["hour"].index60,
    ]
    weights = [0.9, 1.2, 1.4, 1.0]
    z = 0j
    for index, weight in zip(indices, weights):
        z += weight * complex_cycle(index, 60, harmonics=(1, 2, 3))
    return z


def centered_element_projection(vec: np.ndarray) -> np.ndarray:
    normalized = normalize_vec(vec)
    centered = normalized - float(normalized.mean())
    proj = Q @ centered
    scale = float(np.max(np.abs(proj))) or 1.0
    return proj / scale


def personal_line_potentials(profile_pillars: Dict[str, object], birth_num: Dict[str, Optional[int]]) -> np.ndarray:
    phase_g = phase(ganzhi_complex(profile_pillars))
    phase_n = phase(numerology_complex(birth_num))
    vec = element_vector_from_pillars(pillar_tuple_list(profile_pillars))
    projection = centered_element_projection(vec)

    out = []
    for i in range(6):
        phi6 = 2.0 * math.pi * i / 6.0
        phi9 = 2.0 * math.pi * i / 9.0
        value = (
            0.95 * math.cos(phase_g + phi6)
            + 0.65 * math.cos(phase_n + phi9)
            + 0.90 * float(projection[i])
        )
        out.append(value)
    return np.array(out, dtype=float)


def temporal_line_potentials(
    temporal_pillars: Dict[str, object],
    temporal_num: Dict[str, Optional[int]],
    target_dt: datetime,
) -> np.ndarray:
    phase_g = phase(ganzhi_complex(temporal_pillars))
    phase_n = phase(numerology_complex(temporal_num))
    vec = element_vector_from_pillars(pillar_tuple_list(temporal_pillars))
    projection = centered_element_projection(vec)

    jd = dt_to_jdut(target_dt)
    sun_phase = math.radians(sun_longitude(jd))
    moon_phase = math.radians(lunar_phase_angle(jd))

    out = []
    for i in range(6):
        phi6 = 2.0 * math.pi * i / 6.0
        phi9 = 2.0 * math.pi * i / 9.0
        value = (
            0.80 * math.cos(phase_g + phi6)
            + 0.55 * math.cos(phase_n + phi9)
            + 0.75 * math.cos(sun_phase + math.pi * i / 3.0)
            + 0.55 * math.cos(moon_phase + math.pi * i / 6.0)
            + 0.85 * float(projection[i])
        )
        out.append(value)
    return np.array(out, dtype=float)


def combined_line_potentials(
    profile_pillars: Dict[str, object],
    birth_num: Dict[str, Optional[int]],
    temporal_pillars: Dict[str, object],
    temporal_num: Dict[str, Optional[int]],
    target_dt: datetime,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    personal = personal_line_potentials(profile_pillars, birth_num)
    temporal = temporal_line_potentials(temporal_pillars, temporal_num, target_dt)

    phase_g_diff = phase(ganzhi_complex(temporal_pillars)) - phase(ganzhi_complex(profile_pillars))
    phase_n_diff = phase(numerology_complex(temporal_num)) - phase(numerology_complex(birth_num))

    proj_profile = centered_element_projection(element_vector_from_pillars(pillar_tuple_list(profile_pillars)))
    proj_temporal = centered_element_projection(element_vector_from_pillars(pillar_tuple_list(temporal_pillars)))
    delta_proj = proj_temporal - proj_profile

    out = []
    for i in range(6):
        phi6 = 2.0 * math.pi * i / 6.0
        phi9 = 2.0 * math.pi * i / 9.0
        value = (
            0.42 * float(personal[i])
            + 0.58 * float(temporal[i])
            + 0.42 * math.cos(phase_g_diff + phi6)
            + 0.28 * math.cos(phase_n_diff + phi9)
            + 0.45 * float(delta_proj[i])
        )
        out.append(value)

    return np.array(out, dtype=float), personal, temporal


def moving_line_state(
    profile_pillars: Dict[str, object],
    birth_num: Dict[str, Optional[int]],
    temporal_pillars: Dict[str, object],
    temporal_num: Dict[str, Optional[int]],
    target_dt: datetime,
) -> Dict[str, object]:
    combined, personal, temporal = combined_line_potentials(
        profile_pillars, birth_num, temporal_pillars, temporal_num, target_dt
    )
    earlier = target_dt - timedelta(hours=12)
    later = target_dt + timedelta(hours=12)
    before, _, _ = combined_line_potentials(
        profile_pillars, birth_num, temporal_pillars, temporal_num, earlier
    )
    after, _, _ = combined_line_potentials(
        profile_pillars, birth_num, temporal_pillars, temporal_num, later
    )
    derivative = np.abs(after - before) / 2.0

    bits = [1 if value >= 0 else 0 for value in combined]
    moving = []
    for i, value in enumerate(combined):
        tau = 0.24 + 0.14 * min(1.0, float(derivative[i]) / 1.2)
        if abs(float(value)) < tau:
            moving.append(i + 1)

    changed_bits = bits[:]
    for line_no in moving:
        changed_bits[line_no - 1] = 1 - changed_bits[line_no - 1]

    return {
        "combined": combined,
        "personal": personal,
        "temporal": temporal,
        "bits": bits,
        "moving_lines": moving,
        "changed_bits": changed_bits,
        "hex": bits_to_hex(bits),
        "changed_hex": bits_to_hex(changed_bits),
    }


def wuxing_alignment(
    profile_pillars: Dict[str, object],
    temporal_pillars: Dict[str, object],
) -> Tuple[float, np.ndarray, np.ndarray]:
    e_profile = element_vector_from_pillars(pillar_tuple_list(profile_pillars))
    e_temporal = element_vector_from_pillars(pillar_tuple_list(temporal_pillars))
    ep = normalize_vec(e_profile)
    et = normalize_vec(e_temporal)

    day_master = STEM_ELEMENT[profile_pillars["day"].stem]
    balance = 1.0 - 0.5 * float(np.abs(ep - et).sum())
    support = float((et[day_master] + et[PARENT[day_master]]) - (et[CHILD[day_master]] + et[CONTROLLER[day_master]]))

    same = float(np.dot(ep, et))
    generation = sum(float(ep[i] * et[CHILD[i]] + et[i] * ep[CHILD[i]]) for i in range(5)) / 2.0
    control = sum(float(ep[i] * et[CONTROLLED[i]] + et[i] * ep[CONTROLLED[i]]) for i in range(5)) / 2.0

    alignment = 0.35 * (2.0 * balance - 1.0) + 0.25 * support + 0.25 * (generation - same * 0.2) - 0.15 * control
    alignment = max(-1.0, min(1.0, alignment))
    return alignment, ep, et


def numerology_alignment(
    birth_num: Dict[str, Optional[int]],
    temporal_num: Dict[str, Optional[int]],
) -> float:
    pairs = [
        (birth_num["life_path"], temporal_num["personal_year"]),
        (birth_num["attitude"], temporal_num["personal_month"]),
        (birth_num["birthday"], temporal_num["personal_day"]),
    ]
    if birth_num.get("expression") is not None:
        pairs.append((birth_num["expression"], temporal_num["universal_day"]))

    values = []
    for a, b in pairs:
        ra = reduce_num(int(a), keep_master=False)
        rb = reduce_num(int(b), keep_master=False)
        values.append(math.cos(2.0 * math.pi * (ra - rb) / 9.0))

    base = sum(values) / len(values)
    bonus = 0.0
    if birth_num["life_path"] in (11, 22, 33) or temporal_num["personal_day"] in (11, 22, 33):
        bonus += 0.05
    return max(-1.0, min(1.0, base + bonus))


def hex_alignment(
    profile_pillars: Dict[str, object],
    birth_num: Dict[str, Optional[int]],
    temporal_pillars: Dict[str, object],
    temporal_num: Dict[str, Optional[int]],
    target_dt: datetime,
) -> Tuple[float, Dict[str, object]]:
    move_state = moving_line_state(profile_pillars, birth_num, temporal_pillars, temporal_num, target_dt)
    personal_lines = move_state["personal"]
    temporal_lines = move_state["temporal"]

    cosine = float(
        np.dot(personal_lines, temporal_lines)
        / (float(np.linalg.norm(personal_lines)) * float(np.linalg.norm(temporal_lines)) + 1e-9)
    )

    personal_bits = [1 if value >= 0 else 0 for value in personal_lines]
    bit_match = 1.0 - sum(abs(a - b) for a, b in zip(personal_bits, move_state["bits"])) / 6.0

    personal_hex = bits_to_hex(personal_bits)
    temporal_hex = move_state["hex"]

    lower_p = trigram_from_bits(personal_hex["binary_bottom_to_top"][:3])
    upper_p = trigram_from_bits(personal_hex["binary_bottom_to_top"][3:])
    lower_t = trigram_from_bits(temporal_hex["binary_bottom_to_top"][:3])
    upper_t = trigram_from_bits(temporal_hex["binary_bottom_to_top"][3:])

    relation = (
        element_relation_score(TRIGRAM_ELEMENT[lower_p], TRIGRAM_ELEMENT[lower_t])
        + element_relation_score(TRIGRAM_ELEMENT[upper_p], TRIGRAM_ELEMENT[upper_t])
    ) / 2.0

    alignment = 0.55 * cosine + 0.25 * (2.0 * bit_match - 1.0) + 0.20 * relation
    return max(-1.0, min(1.0, alignment)), move_state


def continuum_alignment(
    profile_pillars: Dict[str, object],
    birth_num: Dict[str, Optional[int]],
    target_dt: datetime,
) -> float:
    jd = dt_to_jdut(target_dt)
    sun_phase = math.radians(sun_longitude(jd))
    moon_phase = math.radians(lunar_phase_angle(jd))
    return max(
        -1.0,
        min(
            1.0,
            0.55 * math.cos(sun_phase - phase(ganzhi_complex(profile_pillars)))
            + 0.45 * math.cos(moon_phase - phase(numerology_complex(birth_num))),
        ),
    )


def structure_channels(day_master_element: int, temporal_elements_norm: np.ndarray) -> Dict[str, float]:
    resource = float(temporal_elements_norm[PARENT[day_master_element]])
    companion = float(temporal_elements_norm[day_master_element])
    output = float(temporal_elements_norm[CHILD[day_master_element]])
    wealth = float(temporal_elements_norm[CONTROLLED[day_master_element]])
    authority = float(temporal_elements_norm[CONTROLLER[day_master_element]])
    return {
        "resource": resource + 0.6 * companion,
        "output": output,
        "wealth": wealth,
        "authority": authority,
        "companion": companion,
    }


def score_models(
    profile_pillars: Dict[str, object],
    birth_num: Dict[str, Optional[int]],
    temporal_pillars: Dict[str, object],
    temporal_num: Dict[str, Optional[int]],
    target_dt: datetime,
) -> Dict[str, object]:
    g_align = complex_alignment(ganzhi_complex(profile_pillars), ganzhi_complex(temporal_pillars))
    w_align, ep, et = wuxing_alignment(profile_pillars, temporal_pillars)
    n_align = numerology_alignment(birth_num, temporal_num)
    h_align, move_state = hex_alignment(profile_pillars, birth_num, temporal_pillars, temporal_num, target_dt)
    c_align = continuum_alignment(profile_pillars, birth_num, target_dt)

    day_master = STEM_ELEMENT[profile_pillars["day"].stem]
    balance = 1.0 - 0.5 * float(np.abs(ep - et).sum())
    support = float((et[day_master] + et[PARENT[day_master]]) - (et[CHILD[day_master]] + et[CONTROLLER[day_master]]))
    bazi_energy = 0.62 * (2.0 * balance - 1.0) + 0.38 * support
    bazi_score = sigmoid(2.4 * bazi_energy) * 100.0

    move_penalty = len(move_state["moving_lines"]) / 6.0
    personal_bits = [1 if value >= 0 else 0 for value in move_state["personal"]]
    bit_match = 1.0 - sum(abs(a - b) for a, b in zip(personal_bits, move_state["bits"])) / 6.0
    temporal_hex = move_state["hex"]
    lower = trigram_from_bits(temporal_hex["binary_bottom_to_top"][:3])
    upper = trigram_from_bits(temporal_hex["binary_bottom_to_top"][3:])
    trigram_relation = element_relation_score(TRIGRAM_ELEMENT[lower], TRIGRAM_ELEMENT[upper])
    hex_energy = 0.70 * (2.0 * bit_match - 1.0) + 0.30 * trigram_relation - 0.35 * move_penalty
    hex_score = sigmoid(2.2 * hex_energy) * 100.0

    num_score = sigmoid(2.0 * n_align) * 100.0
    naive_score = (bazi_score + hex_score + num_score) / 3.0

    volatility = min(
        1.0,
        0.55 * move_penalty
        + 0.45
        * float(np.mean(np.abs(np.diff(move_state["combined"]))))
        / (float(np.max(np.abs(move_state["combined"]))) + 1e-6),
    )

    unified_energy = (
        0.28 * g_align
        + 0.22 * w_align
        + 0.14 * n_align
        + 0.22 * h_align
        + 0.14 * c_align
        - 0.16 * volatility
    )
    unified_score = sigmoid(2.7 * unified_energy) * 100.0

    channels = structure_channels(day_master, et)
    channel_scores = {
        key: float(max(0.0, min(100.0, 50.0 + 55.0 * (value - 0.2))))
        for key, value in channels.items()
    }

    return {
        "g_align": float(g_align),
        "w_align": float(w_align),
        "n_align": float(n_align),
        "h_align": float(h_align),
        "continuum": float(c_align),
        "volatility": float(volatility),
        "bazi_score": float(bazi_score),
        "hex_score": float(hex_score),
        "num_score": float(num_score),
        "naive_score": float(naive_score),
        "unified_score": float(unified_score),
        "channels": channel_scores,
        "move_state": move_state,
    }


# -----------------------------
# Engine
# -----------------------------

class UnifiedDestinyEngine:
    def profile_state(
        self,
        birth_dt: datetime,
        name: str = "",
        tz_name: str = "Asia/Taipei",
        longitude_deg: Optional[float] = 121.5654,
        use_true_solar: bool = True,
        zi_rollover: bool = True,
    ) -> Dict[str, object]:
        tz = ZoneInfo(tz_name)
        if birth_dt.tzinfo is None:
            birth_dt = birth_dt.replace(tzinfo=tz)
        else:
            birth_dt = birth_dt.astimezone(tz)

        pillars = pillars_for_datetime(
            birth_dt,
            tz_name=tz_name,
            longitude_deg=longitude_deg,
            use_true_solar=use_true_solar,
            zi_rollover=zi_rollover,
        )
        numbers = numerology_profile(birth_dt, name=name)
        personal_lines = personal_line_potentials(pillars, numbers)
        personal_bits = [1 if value >= 0 else 0 for value in personal_lines]
        personal_hex = bits_to_hex(personal_bits)

        return {
            "birth_dt": birth_dt,
            "name": name,
            "tz_name": tz_name,
            "longitude_deg": longitude_deg,
            "use_true_solar": use_true_solar,
            "zi_rollover": zi_rollover,
            "pillars": pillars,
            "numerology": numbers,
            "personal_lines": personal_lines,
            "personal_hex": personal_hex,
        }

    def forecast(
        self,
        profile: Dict[str, object],
        start_date: date,
        days: int = 7,
    ) -> List[Dict[str, object]]:
        out = []
        birth_dt = profile["birth_dt"]
        tz_name = profile["tz_name"]

        for offset in range(days):
            current_date = start_date + timedelta(days=offset)
            target_dt = datetime.combine(
                current_date,
                datetime.min.time().replace(hour=12, minute=0),
                tzinfo=ZoneInfo(tz_name),
            )

            temporal_pillars = pillars_for_datetime(
                target_dt,
                tz_name=tz_name,
                longitude_deg=profile["longitude_deg"],
                use_true_solar=profile["use_true_solar"],
                zi_rollover=profile["zi_rollover"],
            )
            temporal_num = numerology_profile(birth_dt, target_dt, name=profile["name"])

            scored = score_models(
                profile["pillars"],
                profile["numerology"],
                temporal_pillars,
                temporal_num,
                target_dt,
            )
            move_state = scored["move_state"]

            top_channel_key = max(scored["channels"].items(), key=lambda kv: kv[1])[0]
            top_channel = CHANNEL_LABELS[top_channel_key]

            line_texts = []
            for line_no in move_state["moving_lines"]:
                text = move_state["hex"]["line_texts"].get(str(line_no))
                if text:
                    line_texts.append(f"{line_no}爻：{text}")

            out.append(
                {
                    "date": current_date.isoformat(),
                    "weekday": WEEKDAYS_ZH[target_dt.weekday()],
                    "day_pillar": temporal_pillars["day"].name,
                    "hour_pillar": temporal_pillars["hour"].name,
                    "score_unified": scored["unified_score"],
                    "score_bazi": scored["bazi_score"],
                    "score_hex": scored["hex_score"],
                    "score_num": scored["num_score"],
                    "score_naive": scored["naive_score"],
                    "volatility": scored["volatility"],
                    "label": score_label(scored["unified_score"], scored["volatility"]),
                    "hex_number": move_state["hex"]["number"],
                    "hex_name": move_state["hex"]["name_zh"],
                    "changed_hex_number": move_state["changed_hex"]["number"],
                    "changed_hex_name": move_state["changed_hex"]["name_zh"],
                    "moving_lines": move_state["moving_lines"],
                    "top_channel": top_channel,
                    "channels": {CHANNEL_LABELS[k]: float(v) for k, v in scored["channels"].items()},
                    "line_texts": line_texts,
                    "judgment": move_state["hex"]["judgment"],
                    "image": move_state["hex"]["image"],
                }
            )
        return out

    def summarize_forecast(self, profile: Dict[str, object], records: List[Dict[str, object]]) -> str:
        pillars = profile["pillars"]
        numbers = profile["numerology"]
        personal_hex = profile["personal_hex"]

        strongest = max(records, key=lambda r: r["score_unified"])
        weakest = min(records, key=lambda r: r["score_unified"])

        lines = []
        lines.append(
            f"八字：{pillars['year'].name} / {pillars['month'].name} / {pillars['day'].name} / {pillars['hour'].name}"
        )
        lines.append(
            f"本命卦：{personal_hex['number']:02d} {personal_hex['name_zh']} ；數字：LifePath={numbers['life_path']}  Attitude={numbers['attitude']}  Birthday={numbers['birthday']}"
            + (f"  Expression={numbers['expression']}" if numbers.get("expression") is not None else "")
        )
        lines.append(
            f"最強日：{strongest['date']} {strongest['weekday']}，分數 {strongest['score_unified']:.1f}，主通道 {strongest['top_channel']}，卦象 {strongest['hex_name']}→{strongest['changed_hex_name']}"
        )
        lines.append(
            f"最弱日：{weakest['date']} {weakest['weekday']}，分數 {weakest['score_unified']:.1f}，主通道 {weakest['top_channel']}，卦象 {weakest['hex_name']}→{weakest['changed_hex_name']}"
        )
        lines.append("")
        for rec in records:
            moving = "無" if not rec["moving_lines"] else ",".join(str(x) for x in rec["moving_lines"])
            lines.append(
                f"{rec['date']} {rec['weekday']}｜{rec['label']}｜{rec['score_unified']:.1f}｜主通道 {rec['top_channel']}｜"
                f"本卦 {rec['hex_number']:02d} {rec['hex_name']}｜之卦 {rec['changed_hex_number']:02d} {rec['changed_hex_name']}｜動爻 {moving}"
            )
            for line_text in rec["line_texts"]:
                lines.append(f"    {line_text}")
        return "\n".join(lines)

    def internal_benchmark(
        self,
        sample_size: int = 120,
        start_date: Optional[date] = None,
        days: int = 7,
        tz_name: str = "Asia/Taipei",
        longitude_deg: Optional[float] = 121.5654,
        seed: int = 42,
    ) -> List[Dict[str, object]]:
        if start_date is None:
            start_date = date.today()

        rng = random.Random(seed)
        models = ["bazi", "hex", "num", "naive", "unified"]
        signatures: Dict[str, List[np.ndarray]] = {model: [] for model in models}
        hexagram_variety = {"hex": set(), "unified": set()}

        for _ in range(sample_size):
            y = rng.randint(1970, 2004)
            m = rng.randint(1, 12)
            d = rng.randint(1, calendar.monthrange(y, m)[1])
            hh = rng.randint(0, 23)
            mm = rng.randint(0, 59)

            birth_dt = datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz_name))
            profile = self.profile_state(
                birth_dt,
                name="",
                tz_name=tz_name,
                longitude_deg=longitude_deg,
                use_true_solar=True,
                zi_rollover=True,
            )

            daily = {model: [] for model in models}
            for offset in range(days):
                target_dt = datetime.combine(
                    start_date + timedelta(days=offset),
                    datetime.min.time().replace(hour=12),
                    tzinfo=ZoneInfo(tz_name),
                )
                temporal_pillars = pillars_for_datetime(
                    target_dt,
                    tz_name=tz_name,
                    longitude_deg=longitude_deg,
                    use_true_solar=True,
                    zi_rollover=True,
                )
                temporal_num = numerology_profile(birth_dt, target_dt)
                scored = score_models(
                    profile["pillars"],
                    profile["numerology"],
                    temporal_pillars,
                    temporal_num,
                    target_dt,
                )
                for model in models:
                    daily[model].append(scored[f"{model}_score"])
                hexagram_variety["hex"].add(scored["move_state"]["hex"]["number"])
                hexagram_variety["unified"].add(scored["move_state"]["changed_hex"]["number"])

            for model in models:
                signatures[model].append(np.array(daily[model], dtype=float))

        def pairwise_mean_distance(arrs: List[np.ndarray]) -> float:
            n = len(arrs)
            values: List[float] = []
            for i in range(n):
                ai = arrs[i]
                for j in range(i + 1, n):
                    values.append(float(np.mean(np.abs(ai - arrs[j]))))
            return float(np.mean(values)) if values else 0.0

        rows = []
        for model in models:
            arrs = signatures[model]
            flat = np.concatenate(arrs)
            row = {
                "model": model,
                "label": MODEL_LABELS[model],
                "signature_dispersion": pairwise_mean_distance(arrs),
                "score_std": float(np.std(flat)),
                "distinct_signature_ratio": float(
                    len({tuple(np.round(v, 1).tolist()) for v in arrs}) / sample_size
                ),
                "hexagram_variety": len(hexagram_variety[model]) if model in hexagram_variety else None,
            }
            rows.append(row)

        rows.sort(key=lambda row: row["signature_dispersion"], reverse=True)
        return rows

    def backtest_csv(
        self,
        profile: Dict[str, object],
        csv_path: str,
        date_col: Optional[str] = None,
        outcome_col: Optional[str] = None,
    ) -> Dict[str, object]:
        rows: List[Tuple[datetime, float]] = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV 沒有欄位名稱")
            if date_col is None or outcome_col is None:
                auto_date_col, auto_outcome_col = auto_detect_columns(reader.fieldnames)
                date_col = date_col or auto_date_col
                outcome_col = outcome_col or auto_outcome_col

            if date_col not in reader.fieldnames or outcome_col not in reader.fieldnames:
                raise ValueError(f"找不到欄位: {date_col}, {outcome_col}")

            for row in reader:
                if not row.get(date_col) or not row.get(outcome_col):
                    continue
                try:
                    dt = parse_datetime_guess(row[date_col], profile["tz_name"])
                    outcome = float(row[outcome_col])
                    rows.append((dt, outcome))
                except Exception:
                    continue

        if len(rows) < 3:
            raise ValueError("有效資料少於 3 筆，無法回測")

        predictions = {model: [] for model in ["bazi", "hex", "num", "naive", "unified"]}
        outcomes: List[float] = []

        birth_dt = profile["birth_dt"]
        for dt, outcome in rows:
            temporal_pillars = pillars_for_datetime(
                dt,
                tz_name=profile["tz_name"],
                longitude_deg=profile["longitude_deg"],
                use_true_solar=profile["use_true_solar"],
                zi_rollover=profile["zi_rollover"],
            )
            temporal_num = numerology_profile(birth_dt, dt, name=profile["name"])
            scored = score_models(
                profile["pillars"],
                profile["numerology"],
                temporal_pillars,
                temporal_num,
                dt,
            )
            for model in predictions:
                predictions[model].append(scored[f"{model}_score"])
            outcomes.append(outcome)

        result_rows = []
        for model, values in predictions.items():
            result_rows.append(
                {
                    "model": model,
                    "label": MODEL_LABELS[model],
                    "pearson": pearson_corr(values, outcomes),
                    "spearman": spearman_corr(values, outcomes),
                    "directional_accuracy": directional_accuracy(values, outcomes),
                    "n": len(outcomes),
                }
            )
        result_rows.sort(
            key=lambda row: (
                -abs(row["pearson"]) if not math.isnan(row["pearson"]) else 0.0,
                -abs(row["spearman"]) if not math.isnan(row["spearman"]) else 0.0,
            )
        )
        return {
            "rows": result_rows,
            "date_col": date_col,
            "outcome_col": outcome_col,
            "n": len(outcomes),
        }

    def theory_text(self) -> str:
        return (
            "統一諧振模型（Unified Harmonic Model）\n"
            "===================================\n\n"
            "1. 狀態空間\n"
            "   個人態 x_p 由四柱、五行向量、個人卦線勢、數字向量組成；時間態 x_t 由目標日四柱、時間卦線勢、"
            "太陽黃經、月相角、當日數字向量組成。\n\n"
            "2. 數學核心\n"
            "   (a) 干支與數字以有限循環群的複數角色（Fourier characters）表示：\n"
            "       z_60 = Σ w_k Σ_r (1/r) exp(2π i r g_k / 60)\n"
            "       z_9  = Σ v_k Σ_r (1/r) exp(2π i r n_k / 9)\n\n"
            "   (b) 五行向量由天干顯性元素 + 地支藏干加權構成，再投影到六爻空間：\n"
            "       p = Q · centered(normalized(wuxing))\n\n"
            "   (c) 六爻線勢不是隨機投擲，而是連續場的量化切片：\n"
            "       a_i(t) = 0.42 p_i + 0.58 u_i(t) + 0.42 cos(Δφ_g + 2πi/6)\n"
            "                + 0.28 cos(Δφ_n + 2πi/9) + 0.45 Δp_i\n"
            "       線值 sign(a_i) 給出陰陽；|a_i| 落在低邊際區且導數高時，該爻視為動爻。\n\n"
            "   (d) 統一分數\n"
            "       E = 0.28 A_ganzhi + 0.22 A_wuxing + 0.14 A_num + 0.22 A_hex + 0.14 A_cont - 0.16 V\n"
            "       score = sigmoid(2.7 E) × 100\n\n"
            "3. 創新點\n"
            "   - 把動爻解釋為「線勢接近分岔面」的低邊際狀態，而不是純隨機抽樣。\n"
            "   - 四柱、易卦、數字學全部收進同一能量函數，而非三套規則硬拼接。\n"
            "   - 支援連續天文量（太陽黃經、月相）與離散干支量同時入模。\n\n"
            "4. 驗證\n"
            "   - 內建 benchmark 只驗證結構特性：分數離散度、簽名分散度、卦型覆蓋。\n"
            "   - 真正的外部預測力，必須用使用者自己的結果資料 CSV 回測。\n"
            "   - 因此本程式不宣稱已經用科學實驗證明可以真實預知未來。\n\n"
            "5. CSV 回測格式\n"
            "   至少兩欄：日期欄 + 結果欄。欄名可用 date / datetime / timestamp 與 outcome / score / value。\n"
            "   日期可為 YYYY-MM-DD 或 YYYY-MM-DD HH:MM。\n"
        )


# -----------------------------
# GUI
# -----------------------------

def has_display() -> bool:
    if sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))


class DestinyApp:
    def __init__(self, engine: UnifiedDestinyEngine):
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext

        self.engine = engine
        self.root = tk.Tk()
        self.root.title("統一命理數學模型 / Unified Destiny Harmonic Model")
        self.root.geometry("1500x950")

        self._build_vars()
        self._build_ui()

    def _build_vars(self) -> None:
        tk = self.tk
        today = date.today().isoformat()
        self.birth_var = tk.StringVar(value="1990-01-01 12:34")
        self.name_var = tk.StringVar(value="")
        self.tz_var = tk.StringVar(value="Asia/Taipei")
        self.lon_var = tk.StringVar(value="121.5654")
        self.start_var = tk.StringVar(value=today)
        self.days_var = tk.StringVar(value="7")
        self.true_solar_var = tk.BooleanVar(value=True)
        self.zi_roll_var = tk.BooleanVar(value=True)

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        top = ttk.LabelFrame(outer, text="輸入", padding=8)
        top.pack(fill="x", pady=(0, 8))

        fields = [
            ("出生時間", self.birth_var, 28),
            ("姓名(可空)", self.name_var, 20),
            ("時區", self.tz_var, 18),
            ("經度", self.lon_var, 10),
            ("起算日期", self.start_var, 14),
            ("天數", self.days_var, 6),
        ]
        for idx, (label, var, width) in enumerate(fields):
            ttk.Label(top, text=label).grid(row=0, column=idx * 2, sticky="w", padx=(0, 4), pady=2)
            ttk.Entry(top, textvariable=var, width=width).grid(row=0, column=idx * 2 + 1, sticky="w", padx=(0, 10), pady=2)

        ttk.Checkbutton(top, text="使用真太陽時", variable=self.true_solar_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(top, text="23:00 子初換日", variable=self.zi_roll_var).grid(row=1, column=2, columnspan=2, sticky="w", pady=4)

        ttk.Button(top, text="計算未來一週", command=self.run_forecast).grid(row=1, column=4, padx=4, pady=4, sticky="w")
        ttk.Button(top, text="內建基準測試", command=self.run_benchmark).grid(row=1, column=5, padx=4, pady=4, sticky="w")
        ttk.Button(top, text="載入CSV回測", command=self.run_backtest).grid(row=1, column=6, padx=4, pady=4, sticky="w")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        # Forecast tab
        self.tab_forecast = ttk.Frame(notebook)
        notebook.add(self.tab_forecast, text="未來一週")

        self.forecast_summary = self.scrolledtext.ScrolledText(self.tab_forecast, height=14, wrap="word")
        self.forecast_summary.pack(fill="x", padx=4, pady=4)

        forecast_mid = ttk.Panedwindow(self.tab_forecast, orient="horizontal")
        forecast_mid.pack(fill="both", expand=True, padx=4, pady=4)

        left = ttk.Frame(forecast_mid)
        right = ttk.Frame(forecast_mid)
        forecast_mid.add(left, weight=3)
        forecast_mid.add(right, weight=2)

        columns = ("date", "day_pillar", "score", "label", "hex", "changed", "moving", "channel")
        self.forecast_tree = ttk.Treeview(left, columns=columns, show="headings", height=20)
        headings = {
            "date": "日期",
            "day_pillar": "日柱",
            "score": "統一分數",
            "label": "標記",
            "hex": "本卦",
            "changed": "之卦",
            "moving": "動爻",
            "channel": "主通道",
        }
        widths = {"date": 120, "day_pillar": 90, "score": 95, "label": 90, "hex": 100, "changed": 100, "moving": 80, "channel": 90}
        for col in columns:
            self.forecast_tree.heading(col, text=headings[col])
            self.forecast_tree.column(col, width=widths[col], anchor="center")
        self.forecast_tree.pack(fill="both", expand=True, side="left")
        forecast_scroll = ttk.Scrollbar(left, orient="vertical", command=self.forecast_tree.yview)
        self.forecast_tree.configure(yscrollcommand=forecast_scroll.set)
        forecast_scroll.pack(fill="y", side="right")

        self.details_text = self.scrolledtext.ScrolledText(right, wrap="word")
        self.details_text.pack(fill="both", expand=True)

        self.canvas_widget = None
        self.figure = None

        # Benchmark tab
        self.tab_benchmark = ttk.Frame(notebook)
        notebook.add(self.tab_benchmark, text="基準測試")

        self.benchmark_info = self.scrolledtext.ScrolledText(self.tab_benchmark, height=8, wrap="word")
        self.benchmark_info.pack(fill="x", padx=4, pady=4)

        bench_cols = ("model", "dispersion", "std", "distinct", "variety")
        self.benchmark_tree = ttk.Treeview(self.tab_benchmark, columns=bench_cols, show="headings", height=12)
        bench_head = {
            "model": "模型",
            "dispersion": "簽名分散度",
            "std": "分數標準差",
            "distinct": "不同簽名比率",
            "variety": "卦型覆蓋",
        }
        for col in bench_cols:
            self.benchmark_tree.heading(col, text=bench_head[col])
            self.benchmark_tree.column(col, width=160, anchor="center")
        self.benchmark_tree.pack(fill="both", expand=True, padx=4, pady=4)

        # Backtest tab
        self.tab_backtest = ttk.Frame(notebook)
        notebook.add(self.tab_backtest, text="CSV回測")

        self.backtest_info = self.scrolledtext.ScrolledText(self.tab_backtest, height=8, wrap="word")
        self.backtest_info.pack(fill="x", padx=4, pady=4)
        self.backtest_info.insert(
            "1.0",
            "CSV 至少要有兩欄：日期欄 + 結果欄。\n"
            "常見欄名可用 date / datetime / timestamp 與 outcome / score / value。\n"
            "結果欄必須可轉成浮點數。\n",
        )

        back_cols = ("model", "pearson", "spearman", "directional", "n")
        self.backtest_tree = ttk.Treeview(self.tab_backtest, columns=back_cols, show="headings", height=12)
        back_head = {
            "model": "模型",
            "pearson": "Pearson",
            "spearman": "Spearman",
            "directional": "方向命中率",
            "n": "樣本數",
        }
        for col in back_cols:
            self.backtest_tree.heading(col, text=back_head[col])
            self.backtest_tree.column(col, width=160, anchor="center")
        self.backtest_tree.pack(fill="both", expand=True, padx=4, pady=4)

        # Theory tab
        self.tab_theory = ttk.Frame(notebook)
        notebook.add(self.tab_theory, text="理論")
        theory = self.scrolledtext.ScrolledText(self.tab_theory, wrap="word")
        theory.pack(fill="both", expand=True, padx=4, pady=4)
        theory.insert("1.0", self.engine.theory_text())
        theory.configure(state="disabled")

    def _build_profile_from_inputs(self) -> Dict[str, object]:
        tz_name = self.tz_var.get().strip() or "Asia/Taipei"
        birth_dt = parse_datetime_guess(self.birth_var.get(), tz_name)
        name = self.name_var.get().strip()
        longitude_text = self.lon_var.get().strip()
        longitude = float(longitude_text) if longitude_text else None
        return self.engine.profile_state(
            birth_dt=birth_dt,
            name=name,
            tz_name=tz_name,
            longitude_deg=longitude,
            use_true_solar=self.true_solar_var.get(),
            zi_rollover=self.zi_roll_var.get(),
        )

    def _selected_start_date(self, tz_name: str) -> date:
        return parse_datetime_guess(self.start_var.get(), tz_name).date()

    def _selected_days(self) -> int:
        value = int(self.days_var.get().strip())
        if value <= 0:
            raise ValueError("天數必須大於 0")
        return value

    def _clear_tree(self, tree) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def run_forecast(self) -> None:
        try:
            profile = self._build_profile_from_inputs()
            start = self._selected_start_date(profile["tz_name"])
            days = self._selected_days()
            records = self.engine.forecast(profile, start, days)
            summary = self.engine.summarize_forecast(profile, records)

            self.forecast_summary.delete("1.0", "end")
            self.forecast_summary.insert("1.0", summary)

            self._clear_tree(self.forecast_tree)
            for rec in records:
                moving = "無" if not rec["moving_lines"] else ",".join(str(x) for x in rec["moving_lines"])
                self.forecast_tree.insert(
                    "",
                    "end",
                    values=(
                        f"{rec['date']} {rec['weekday']}",
                        rec["day_pillar"],
                        f"{rec['score_unified']:.1f}",
                        rec["label"],
                        f"{rec['hex_number']:02d} {rec['hex_name']}",
                        f"{rec['changed_hex_number']:02d} {rec['changed_hex_name']}",
                        moving,
                        rec["top_channel"],
                    ),
                )

            detail_lines = []
            for rec in records:
                detail_lines.append(
                    f"{rec['date']} {rec['weekday']}｜{rec['label']}｜{rec['score_unified']:.1f}\n"
                    f"  本卦 {rec['hex_number']:02d} {rec['hex_name']} → 之卦 {rec['changed_hex_number']:02d} {rec['changed_hex_name']}\n"
                    f"  Judgment: {rec['judgment']}\n"
                )
                for line_text in rec["line_texts"]:
                    detail_lines.append(f"  {line_text}\n")
                detail_lines.append("\n")
            self.details_text.delete("1.0", "end")
            self.details_text.insert("1.0", "".join(detail_lines))

            self._draw_chart(records)
        except Exception as exc:
            self.messagebox.showerror("錯誤", str(exc))

    def _draw_chart(self, records: List[Dict[str, object]]) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception:
            return

        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()
            self.canvas_widget = None

        dates = [rec["date"][5:] for rec in records]
        scores = [rec["score_unified"] for rec in records]

        fig = plt.Figure(figsize=(5.2, 3.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(dates, scores, marker="o")
        ax.set_title("未來一週統一分數")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 100)
        ax.grid(True)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # re-import for type check

        canvas = FigureCanvasTkAgg(fig, master=self.tab_forecast)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=4, pady=(0, 8))
        self.canvas_widget = canvas
        self.figure = fig

    def run_benchmark(self) -> None:
        try:
            rows = self.engine.internal_benchmark(
                sample_size=120,
                start_date=parse_datetime_guess(self.start_var.get(), self.tz_var.get().strip() or "Asia/Taipei").date(),
                days=self._selected_days(),
                tz_name=self.tz_var.get().strip() or "Asia/Taipei",
                longitude_deg=float(self.lon_var.get()) if self.lon_var.get().strip() else None,
                seed=42,
            )
            self._clear_tree(self.benchmark_tree)
            for row in rows:
                variety = "—" if row["hexagram_variety"] is None else str(row["hexagram_variety"])
                self.benchmark_tree.insert(
                    "",
                    "end",
                    values=(
                        row["label"],
                        f"{row['signature_dispersion']:.3f}",
                        f"{row['score_std']:.3f}",
                        f"{row['distinct_signature_ratio']:.3f}",
                        variety,
                    ),
                )

            self.benchmark_info.delete("1.0", "end")
            self.benchmark_info.insert(
                "1.0",
                "這裡不是外部真實世界預測力證明，而是內部結構 benchmark。\n"
                "解讀方式：\n"
                "1. 簽名分散度越高，代表不同出生資料得到的整週軌跡越不容易塌縮成相似模板。\n"
                "2. 分數標準差越高，代表模型在 0-100 區間的資訊使用更充分，而不是全部擠在中間。\n"
                "3. 不同簽名比率越高，代表週級輸出模式更少重複。\n"
                "4. 卦型覆蓋越高，代表可到達的卦象狀態更廣。\n"
                "真正的可驗證預測力，請改用 CSV 回測。\n",
            )
        except Exception as exc:
            self.messagebox.showerror("錯誤", str(exc))

    def run_backtest(self) -> None:
        try:
            csv_path = self.filedialog.askopenfilename(
                title="選擇 CSV",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            )
            if not csv_path:
                return

            profile = self._build_profile_from_inputs()
            result = self.engine.backtest_csv(profile, csv_path)

            self._clear_tree(self.backtest_tree)
            for row in result["rows"]:
                pearson = "nan" if math.isnan(row["pearson"]) else f"{row['pearson']:.4f}"
                spearman = "nan" if math.isnan(row["spearman"]) else f"{row['spearman']:.4f}"
                direction = "nan" if math.isnan(row["directional_accuracy"]) else f"{row['directional_accuracy']:.4f}"
                self.backtest_tree.insert(
                    "",
                    "end",
                    values=(row["label"], pearson, spearman, direction, row["n"]),
                )

            self.backtest_info.delete("1.0", "end")
            self.backtest_info.insert(
                "1.0",
                f"回測完成。\n日期欄：{result['date_col']}\n結果欄：{result['outcome_col']}\n樣本數：{result['n']}\n"
                "排序依 |Pearson| 與 |Spearman| 由高到低。\n",
            )
        except Exception as exc:
            self.messagebox.showerror("錯誤", str(exc))

    def run(self) -> None:
        self.root.mainloop()


# -----------------------------
# CLI
# -----------------------------

def cli_forecast_text(records: List[Dict[str, object]]) -> str:
    lines = []
    for rec in records:
        moving = "無" if not rec["moving_lines"] else ",".join(str(x) for x in rec["moving_lines"])
        lines.append(
            f"{rec['date']} {rec['weekday']} | {rec['score_unified']:.1f} | {rec['label']} | "
            f"{rec['hex_number']:02d}{rec['hex_name']} -> {rec['changed_hex_number']:02d}{rec['changed_hex_name']} | 動爻 {moving} | 主通道 {rec['top_channel']}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Destiny Harmonic Model")
    parser.add_argument("--nogui", action="store_true", help="不要啟動 GUI")
    parser.add_argument("--demo", action="store_true", help="輸出 CLI 預測結果")
    parser.add_argument("--benchmark", action="store_true", help="輸出內建 benchmark")
    parser.add_argument("--backtest", type=str, default="", help="回測 CSV 路徑")
    parser.add_argument("--birth", type=str, default="1990-01-01 12:34", help="出生時間，例如 1990-01-01 12:34")
    parser.add_argument("--name", type=str, default="", help="姓名，可空")
    parser.add_argument("--timezone", type=str, default="Asia/Taipei", help="IANA 時區")
    parser.add_argument("--longitude", type=str, default="121.5654", help="經度，可空")
    parser.add_argument("--start-date", type=str, default=date.today().isoformat(), help="起算日期")
    parser.add_argument("--days", type=int, default=7, help="天數")
    parser.add_argument("--no-true-solar", action="store_true", help="停用真太陽時")
    parser.add_argument("--no-zi-rollover", action="store_true", help="停用 23:00 換日")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    engine = UnifiedDestinyEngine()

    want_gui = not args.nogui and not args.demo and not args.benchmark and not args.backtest
    if want_gui and has_display():
        app = DestinyApp(engine)
        app.run()
        return

    tz_name = args.timezone
    longitude = float(args.longitude) if args.longitude.strip() else None
    profile = engine.profile_state(
        birth_dt=parse_datetime_guess(args.birth, tz_name),
        name=args.name,
        tz_name=tz_name,
        longitude_deg=longitude,
        use_true_solar=not args.no_true_solar,
        zi_rollover=not args.no_zi_rollover,
    )

    if args.benchmark:
        rows = engine.internal_benchmark(
            sample_size=120,
            start_date=parse_datetime_guess(args.start_date, tz_name).date(),
            days=args.days,
            tz_name=tz_name,
            longitude_deg=longitude,
            seed=42,
        )
        print("內建 benchmark（結構指標，不是真實世界預測證明）")
        print("=" * 72)
        for row in rows:
            variety = "—" if row["hexagram_variety"] is None else str(row["hexagram_variety"])
            print(
                f"{row['label']:<12}  dispersion={row['signature_dispersion']:.3f}  "
                f"std={row['score_std']:.3f}  distinct={row['distinct_signature_ratio']:.3f}  variety={variety}"
            )

    if args.demo or (not want_gui and not args.backtest and not args.benchmark):
        start = parse_datetime_guess(args.start_date, tz_name).date()
        records = engine.forecast(profile, start, args.days)
        print(engine.summarize_forecast(profile, records))
        print("\n簡表")
        print("-" * 72)
        print(cli_forecast_text(records))

    if args.backtest:
        result = engine.backtest_csv(profile, args.backtest)
        print("\nCSV 回測")
        print("=" * 72)
        print(f"日期欄: {result['date_col']} | 結果欄: {result['outcome_col']} | 樣本數: {result['n']}")
        for row in result["rows"]:
            pearson = "nan" if math.isnan(row["pearson"]) else f"{row['pearson']:.4f}"
            spearman = "nan" if math.isnan(row["spearman"]) else f"{row['spearman']:.4f}"
            direct = "nan" if math.isnan(row["directional_accuracy"]) else f"{row['directional_accuracy']:.4f}"
            print(f"{row['label']:<12}  pearson={pearson}  spearman={spearman}  directional={direct}")


if __name__ == "__main__":
    main()
