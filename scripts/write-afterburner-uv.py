#!/usr/bin/env python3
"""Rewrite an MSI Afterburner 4.6.x VF curve: flatten from --mv (mV) to --mhz.

Close Afterburner before writing. The UI rewrites the F column on launch;
trust the sliders and nvidia-smi dmon, not the file's F field.

  python3 scripts/write-afterburner-uv.py --cfg /path/to/VEN_10DE....cfg \\
      --mv 800 --mhz 1800 --mem 1250 --power 112
"""
from __future__ import annotations

import argparse
import re
import shutil
import struct
from datetime import datetime
from pathlib import Path

# No board's settled OC is the default. Passing only --cfg must not arm +1250/1800.
DEFAULT_V = 800.0

WIN_PROFILES = Path("/mnt/c/Program Files (x86)/MSI Afterburner/Profiles")


def discover_cfg() -> Path | None:
    if not WIN_PROFILES.is_dir():
        return None
    hits = sorted(WIN_PROFILES.glob("VEN_10DE*.cfg"))
    return hits[0] if len(hits) == 1 else None


def section_span(text: str, name: str) -> tuple[int, int, str]:
    start = text.find(f"[{name}]")
    if start < 0:
        raise SystemExit(f"missing [{name}]")
    nxt = re.search(r"\r?\n\[", text[start + 1 :])
    end = start + 1 + nxt.start() if nxt else len(text)
    return start, end, text[start:end]


def extract_curve(sec: str) -> tuple[bytes, int, int]:
    idx = sec.find("VFCurve=")
    if idx < 0:
        raise SystemExit("no VFCurve")
    m = re.search(r"\r?\n[A-Za-z][A-Za-z0-9]*=", sec[idx + 8 :])
    if not m:
        raise SystemExit("no key after VFCurve")
    hexpart = sec[idx + 8 : idx + 8 + m.start()]
    hexstr = "".join(c for c in hexpart if c in "0123456789abcdefABCDEF")
    return bytes.fromhex(hexstr), idx, idx + 8 + m.start()


def build_curve(stock: bytes, live: bytes, target_v: float, target_f: float) -> bytes:
    if len(stock) != len(live):
        raise SystemExit(f"curve length mismatch {len(stock)} vs {len(live)}")
    ver, npts, _z = struct.unpack_from("<III", stock, 0)
    if (ver, npts) != (0x20000, 128):
        raise SystemExit(f"unexpected header {(ver, npts)} — Afterburner Format=2 expected")
    new = bytearray(live)
    for i in range(128):
        off = 12 + i * 12
        v, _fs, _us = struct.unpack_from("<fff", live, off)
        vd, fd, _ud = struct.unpack_from("<fff", stock, off)
        if abs(v - vd) > 1e-2:
            raise SystemExit(f"voltage grid mismatch at {i}: {v} vs {vd}")
        if v + 1e-3 >= target_v:
            struct.pack_into("<fff", new, off, vd, target_f, target_f - fd)
        else:
            struct.pack_into("<fff", new, off, vd, fd, 0.0)
    return bytes(new)


def patch_section(sec: str, hex_new: str, mem_boost: int, power: int) -> str:
    sec, _ = re.subn(r"CoreClkBoost=-?\d+", "CoreClkBoost=0", sec, count=1)
    sec, _ = re.subn(r"MemClkBoost=-?\d+", f"MemClkBoost={mem_boost}", sec, count=1)
    sec, _ = re.subn(r"PowerLimit=\d+", f"PowerLimit={power}", sec, count=1)
    idx = sec.find("VFCurve=")
    m = re.search(r"\r?\n[A-Za-z][A-Za-z0-9]*=", sec[idx + 8 :])
    return sec[:idx] + "VFCurve=" + hex_new + sec[idx + 8 + m.start() :]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cfg", type=Path, default=None, help="Afterburner profile .cfg (VEN_10DE…)")
    ap.add_argument("--mv", type=float, default=DEFAULT_V)
    ap.add_argument("--mhz", type=float, required=True, help="flat frequency from --mv upward (MHz)")
    ap.add_argument("--mem", type=int, required=True, help="memory slider offset, MHz (0 = stock)")
    ap.add_argument("--power", type=int, required=True, help="power limit percent (e.g. 100 or 112)")
    args = ap.parse_args()
    src = args.cfg or discover_cfg()
    if src is None:
        raise SystemExit(
            "pass --cfg /mnt/c/Program Files (x86)/MSI Afterburner/Profiles/VEN_10DE….cfg"
        )
    mem_boost = args.mem * 1000
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = src.with_name(src.name + f".bak-{ts}")
    shutil.copy2(src, bak)
    text = src.read_bytes().decode("utf-8")
    _, _, defsec = section_span(text, "Defaults")
    stock, _, _ = extract_curve(defsec)
    s_start, s_end, ssec = section_span(text, "Startup")
    scurve, _, _ = extract_curve(ssec)
    new = build_curve(stock, scurve, args.mv, args.mhz)
    hex_new = new.hex().upper()
    new_s = patch_section(ssec, hex_new, mem_boost, args.power)
    p_start, p_end, psec = section_span(text, "Profile1")
    new_p = patch_section(psec, hex_new, mem_boost, args.power)
    out = text[:s_start] + new_s + text[s_end:p_start] + new_p + text[p_end:]
    tmp = src.with_suffix(src.suffix + ".tmpwrite")
    tmp.write_bytes(out.encode("utf-8"))
    tmp.replace(src)
    print(f"backup {bak}")
    print(f"wrote  {src}")
    print(f"curve  V>={args.mv:.0f} -> {args.mhz:.0f}; mem +{args.mem}; PL {args.power}")


if __name__ == "__main__":
    main()
