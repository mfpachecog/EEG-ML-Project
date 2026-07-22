# NOTES — held-out download + CAR/bad-channel fix (autonomous session, 2026-07-22)

Working note left by the delegated build session. Nothing here is committed. The
Orchestrator will fold anything worth keeping into the formal Spanish docs
(findings/logs) later. Bilingual/English on purpose (working layer).

---

## TASK 1 — CAR / bad-channel contamination fix (findings 021 §3.3, 022 §3)

### What was changed in `development/src/preprocessing.py`
1. **New function `detect_flat_channels(raw, flat_fraction=FLAT_STD_FRACTION) -> list[str]`**
   (placed just above `rereference_car`). Computes per-channel `std` over the WHOLE
   segment, compares each against `flat_fraction * median(channel stds)`, returns the
   names of channels below threshold. Reuses the existing `FLAT_STD_FRACTION = 1e-3`
   constant (a truly disconnected channel is constant -> std ~0, so the relative
   threshold separates it cleanly from channels with real signal; no distinct
   constant needed).
2. **Wired into `preprocess_patient`'s per-segment loop**, AFTER `filter_raw` +
   `resample_raw` and BEFORE `rereference_car`:
   ```python
   raw.info["bads"] = detect_flat_channels(raw)
   rereference_car(raw)
   ```
3. **Docstrings updated** (module step-4 line + `rereference_car`) to note that CAR now
   excludes `info['bads']` from the average. The documented 6-step order is unchanged —
   this is a refinement *within* step 4, not a reordering.

### MNE bads-in-average behavior — VERIFIED EMPIRICALLY (mne 1.11.0), not assumed
Synthetic `RawArray` (4 ch, one constant/dead, marked in `info['bads']`),
`set_eeg_reference(ref_channels="average", projection=False)`:
- Good channel `B` came out as `B - mean(good channels only)` -> **bad channel IS
  excluded from the average**.
- The dead channel stayed exactly `0.0` -> **bad channels are left untouched** (not
  re-referenced), so they stay flat and the paso-6 `is_flat` criterion can reject them.

=> The default `ref_channels="average"` already does the right thing once channels are
in `info['bads']`. The explicit-good-list approach was NOT needed. (Approach 1 chosen.)

### End-to-end synthetic demo (19 ch, one dead channel across whole segment)
| metric                         | ORIGINAL (no fix) | FIXED |
|--------------------------------|-------------------|-------|
| channels marked bad            | []                | ['F4'] |
| std of dead channel AFTER CAR  | 0.2222 (revived!) | 0.0000 (stays flat) |
| n_flat (epochs caught)         | 0                 | 6 |
| n_kept                         | 5 (contaminated)  | 0 |
| mean contamination of a GOOD channel (|orig-fixed|) | — | 0.0098 |

Confirms exactly the bug the findings describe: without the fix the dead channel is
"revived" by CAR (std>0), escapes `is_flat`, and its inverted-average noise is baked
into all surviving epochs' good channels. With the fix it is excluded from the average,
stays flat, and paso 6 rejects those epochs.

### Before/after on REAL patients (preprocess_patient, default cap, NOT saved to disk)
Original path reproduced by monkeypatching `detect_flat_channels -> []` (the only
runtime change is the `raw.info['bads']=` line, so bads=[] == pre-fix exactly).

| patient | metric | ORIGINAL | FIXED |
|---------|--------|----------|-------|
| 0424 | n_input / n_flat / n_extreme / n_kept | 5760 / 1080 / 0 / 4680 | same |
| 0424 | bad-channel marks | 0 | 0 (detected: []) |
| 0319 | n_input / n_flat / n_extreme / n_kept | 5020 / 912 / 926 / 4094 | same |
| 0319 | bad-channel marks | 0 | 0 (detected: []) |
| 0284 | n_input / n_flat / n_extreme / n_kept | 4238 / 0 / 97 / 4141 | same |
| 0284 | bad-channel marks | 0 | 0 (detected: []) |

**Interpretation:** identical numbers = **no regression**. 0424/0319 have ZERO
segment-persistent dead channels -> confirms finding 022's read that their high flat
counts are *intermittent global-dropout* episodes (caught per-epoch), NOT single dead
electrodes. 0284 (clean) is unchanged, as expected. The fix only bites on segments with
a channel dead across the whole segment (shown in the synthetic demo).

### Should the full batch be re-run? (decision left to the user)
Because these 3 patients showed identical stats, the fix would **not** change the
current 34k-epoch dataset *for them*. But other dev patients were not checked here, and
any segment with a persistently-dead electrode would change (those epochs would now be
correctly rejected instead of silently contaminated). **Recommendation:** a full
`run_batch()` re-run is *cheap insurance* and would make the dataset provably free of
the CAR-revival artifact — but it OVERWRITES `data_processed/` and the documented 34k
numbers (finding 020), so it's a documentation-consistency decision for the user, not a
correctness emergency. NOT re-run here (hard constraint: never touch the real batch).

---

## TASK 2 — held-out test set download

### Chosen 15 NEW patient IDs (disjoint from the 20-patient dev cohort — verified)
- **Good (8):** 0303 0306 0312 0313 0316 0320 0326 0328
- **Poor (7):** 0349 0353 0356 0358 0359 0360 0364

Balance ~8/7 Good/Poor. Outcomes read from each patient's `.txt` on PhysioNet
(no root-level clinical CSV exists in i-care 2.1 — only LICENSE.txt, RECORDS,
SHA256SUMS.txt). All 15 verified to exist in
`https://physionet.org/files/i-care/2.1/training/RECORDS` (607 patients total) and
confirmed NOT already on disk and NOT in the dev cohort.

### Disk reasoning -> 15 (not 10)
- `du -sh patients_data_raw` = **65G** for the 20 dev patients (~3.25 GB/patient).
- `df -h` on /home = **176G available**.
- 15 new patients ~ 49 GB << 176 GB. Comfortable, so went with the top of the 10-15
  range for maximum held-out statistical power.

### How it was launched (detached, survives session end)
```bash
cd patients_data_raw
nohup bash script_holdout.sh > holdout_download.log 2>&1 &
disown
```
- Script: `patients_data_raw/script_holdout.sh` (same wget pattern as script.sh:
  `wget -r -N -c -np ".../training/${patient}/"`, run from inside `patients_data_raw/`
  so files land under `physionet.org/files/i-care/2.1/training/<pid>/`, matching
  `DATA_DIR` in `config.py`).
- Log: `patients_data_raw/holdout_download.log`.
- Confirmed running at launch (bash + wget on 0303 alive, log streaming).
- The original `patients_data_raw/script.sh` was LEFT UNTOUCHED (historical artifact;
  all its IDs are already-present dev patients, NOT held-out).

### After download completes — quick checks to do
- `du -sh` the 15 new folders; confirm each has EEG + `.txt`.
- Re-confirm outcomes from the on-disk `.txt` (Good/Poor) match the table above.
- These are FROZEN: touched only once at Phase 5 with the final model.
