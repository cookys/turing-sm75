# Clocks on a 2080 Ti (Afterburner)

Decision law: [`method.md`](method.md). Example scores: [`results.md`](results.md).

This is the **how to sweep**, not a promise that +285 / +1250 is right for your board.

## Tools

- Windows: MSI Afterburner 4.6.x. Curve editor **X = millivolts, Y = MHz**.
- WSL sees the same driver clocks. You do not overclock from `nvidia-smi -pl` inside WSL (no permission).
- Writer: `scripts/write-afterburner-uv.py` (Afterburner **closed**). Pass `--cfg` to your `VEN_10DE&DEV_1E07….cfg` under `MSI Afterburner/Profiles`.
- Format 2 VFCurve: 3224 bytes, 128 points × `{float V_mV, float F_MHz, float U}`. Voltage grid `450 + i×6.25` mV. `U ≈ target − stock F`.

On launch Afterburner turns a flat-from-knee curve into a **core slider offset** and rewrites F, leaving U. The file no longer looks flat. That does not mean it failed. Trust sliders + `nvidia-smi dmon` under load.

## Walls we actually hit

| Wall | Example 2080 Ti |
|---|---|
| VBIOS power | `max_limit` **280 W** (stock 250). Not 325 W. |
| Thermal | ~84–89 °C. 27B load peaked ~79 °C — **not** the limiter. |
| Power cap | Prefill hits `SW Power Cap` immediately. Decode's best runs sat **under** 280 W with locked `pclk`. |
| Stability | Core +315 / +345 → CUDA abort in llama.cpp (`l2_norm` / `rope` / sync). Not heat, not the 280 W cap. |

## Sweep order

1. Confirm boost (`pclk` not 1100, not ncu-pinned 1545).
2. Memory slider, one step at a time. Kill if not monotonic.
3. Core / VF curve. Small steps. Abort = revert.
4. Never memory+core in one change.
5. Re-run the four-grid **and** long-prompt MTP. They can flip sign.

ncu from an elevated Windows prompt defaults to clock-control and pins core at stock Boost (1545). Memory OC remains. Fix: `--clock-control none`. If already pinned: **reboot Windows**. Do not install a Linux `nvidia-driver*` in WSL to get ncu.

## Example settled sliders (one board, 2026-08-18)

Core **+285**, mem **+1250** (load `mclk` ~8250), power 112% → 280 W, voltage slider 0. Intent: 800 mV and up held at 1800 MHz. Afterburner displayed +285; load `pclk` ~1890. Same sliders, tg128 typically **34.6–35.6**. `+1300` mem did not beat +1250. `+1200` was slower.
