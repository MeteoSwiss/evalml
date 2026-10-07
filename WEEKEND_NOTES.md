# Weekend run: add ICON-CH1/CH2 baselines to the key figures

Started 2026-09-25. Working notes for the unattended run; decisions and problems are
logged here as they happen.

## Summary for Monday (read this first)

**Status: done.** All requested figures exist under `output/with_icon/`. Nothing
existing was overwritten. **I could not look at any of the images** (no image viewer in
husk), so please check them before they go on slides.

New figures:
- `output/with_icon/skill/skill_overview.png`: Multistep and ICON-CH2 against Varda-single,
  lead 0-120 h, with PS. Off-scale medians (CH2 PS) are marked with an arrow and their value.
- `output/with_icon/skill/skill_overview_0-33h.png`: the same plus ICON-CH1, lead 0-33 h.
- `output/with_icon/gap_robustness/paired_gap_overview.png`: two rows, Varda minus Multistep
  (identical to the existing figure's numbers) and Varda minus ICON-CH2, own y axis per row.
- `output/with_icon/spectra_bandpass/figs_FRAC/order4/all_seasons_params_FRAC_models.png`:
  Varda, Multistep, ICON-CH2, now with PS. A `figs_frac` (vs stations) version exists too.
- `output/with_icon/joint_variance_skill/timing_vs_variance.png` + 8 panels in
  `timing_vs_variance/`. x axis extends to ~2.25 because ICON-CH2 retains up to 2.2x REA-L's
  variance at single stations (median 0.83).

Main results (details and tables: FINDINGS.md section 14, marked as not yet reviewed):
1. ICON-CH2 retains 61-109 % of REA-L's 3-6 h variance (ML models 7-26 %), at every station.
2. ICON-CH2's timing correlation is 0.01-0.17: no better, mostly worse, than the ML models.
   So low timing skill at 24-97 h lead looks like a predictability limit, not an ML artefact.
3. By MSE, ICON-CH2 is clearly worse than Varda-single (double penalty). ICON-CH1 beats Varda
   only for SP_10M and PS at 0-33 h, where it shares REA-L's model climate.
4. Caveat everywhere: REA-L is ICON-CH1-based and flatters ICON. CH2's PS skill is
   meaningless (~5 hPa orography offset).

Decisions I took on my own, worth a second look:
- Read ICON GRIB with eccodes directly instead of `load_forecast_data` (husk blocks the
  grid-coordinate lookup, see log). Checked against REA-L at one init; CH1 cells equal the
  REA-L cells.
- Control member only. PS kept in the skill figures despite the offset, with a note.
- 0-33 h skill requires 27 of 34 valid steps per (init, station) cell.
- Plot fixes that only affect multi-model figures: own y axis per row in the gap figure,
  data-driven x range in the timing figure, arrows for off-scale skill medians.
- FINDINGS.md: new section 14, bottom-line point 7, a note in section 12. All marked NEW.

Loose ends:
- SLURM test job 5397150 (preemptible) may still be queued; husk could not cancel it.
  `scancel 5397150` if you like.
- Untracked files for git: `.uvcache/` (uv cache for husk, can be deleted), `output/icon_test/`
  (test, logs, driver scripts), `resources/icon-*`, `WEEKEND_NOTES.md`. Nothing committed.

## Task (from Louis)

Make NEW copies (never overwrite) of these figures including the ICON baselines,
under `output/with_icon/`, mirroring the current paths:

- `output/skill/skill_overview.png`: Varda-single stays the reference; add PS.
- `output/gap_robustness/paired_gap_overview.png`
- `output/spectra_bandpass/figs_FRAC/order4/all_seasons_params_FRAC_models.png` (add PS)
- `output/joint_variance_skill/timing_vs_variance.png` and the 8 single panels in
  `output/joint_variance_skill/timing_vs_variance/`

Louis has authorised decisions by my own judgement for this run (CLAUDE.md's
"keep the user in the loop" is suspended for it).

## Decisions already agreed

- ICON-CH2 (0-120 h hourly, 6-hourly inits) goes into all figures.
- ICON-CH1 only runs 0-33 h, so it cannot enter the variance/gap/timing figures
  (they use lead 24-97 h). It goes into the skill figure ONLY, as a separate
  0-33 h panel where all models are scored over the same 0-33 h window.
- Control member (000), matching the `-CTRL` labels in the configs.
- All models on the same inits per figure; report any sample shrinkage.
- Caveat on every figure: truth is REA-L, which is ICON-based and flatters ICON.
- Scripts get options for models and output dir; with no options they must
  reproduce today's CSVs exactly (check against existing outputs).

## Data

- ICON-CH2: `/store_new/mch/msopr/osm/ICON-CH2-EPS/FCST24/{yymmddHH}_7xx/grib/i2eff{DD}{HH}0000_000`
- ICON-CH1: `/store_new/mch/msopr/osm/ICON-CH1-EPS/FCST24/{yymmddHH}_6xx/grib/i1eff...`
- Loader: `load_forecast_data` / `_load_icon_baseline_from_grib` in `src/data_input/__init__.py`.
  ICON constants (orography) copied to `resources/icon-constants/`; the loader looks in
  `~/.cache/evalml-lapse`, so point `data_input._ICON_CONST_CACHE` there from the script.
- Station mapping: map by lat/lon for both CH1 and CH2 (`src/verification/spatial.py:106`,
  `map_forecast_to_truth`), do not assume REA-L grid order.
- Station classes: `resources/spaziurat.lf_2.6.1.tar.gz` (DONE 2026-09-25: `Path.home()` paths replaced in
  gap_robustness.py, joint_variance_skill.py, spectra_by_station.py, skill_by_station_class.py).
- Output point series: `output/points/{season}/ICON-CH{1,2}-CTRL/<init>.nc`, same format as
  the existing sources.

## Environment notes (husk)

- Run Python as `UV_CACHE_DIR=<project>/.uvcache uv run --no-sync python ...`
  (the uv cache on scratch is read-only inside husk; rdata is installed in the venv).
- SLURM jobs are refused when submitted from under /users (home). Hence the move of the
  project back to scratch on 2026-09-25, into the older checkout
  `/scratch/mch/lfrey/2025_SEN_eval_ML/software/evalml/2026_09_08_assess_multistep/evalml`
  (a git repo; home files were rsynced over it with --update, nothing deleted). Allowed partition was only `preemptible`, account s83;
  Louis may add `postproc`. Test job 5397150 submitted to `preemptible` on 2026-09-25, result
  still to be checked. If jobs fail, fall back to the login node with at most 4 processes. Set env vars INSIDE job scripts (husk does not forward them).
- Include `#SBATCH --exclude=nid[001225-001230]` on postproc.

## Plan

1. One-init test for CH2 and CH1 inside husk: load, coordinates, station mapping. Stop and
   document if it fails.
2. Extract point series for all 82 inits (SLURM array if allowed, else login node, max 4 procs).
3. Generalise scripts (models + output dir options); verify defaults reproduce existing CSVs.
4. Produce the new figures under `output/with_icon/`.
5. Record results, decisions and problems here; add a clearly marked update to FINDINGS.md.

## Log

### 2026-09-25 evening

- **ICON loading via `load_forecast_data` fails inside husk.** earthkit asks `eckit::geo` for the
  unstructured-grid coordinates; eckit expands `~/.cache/eckit/geo`, the passwd lookup fails in
  the sandbox (`Assertion failed: pw ... expandTilde`), and it would then download the grid from
  `sites.ecmwf.int` (not on the allowlist). **Workaround:** new script
  `workflow/scripts/extract_points_icon.py` reads the ICON GRIB directly with eccodes and takes
  cell coordinates (`tlat`/`tlon`) from `resources/icon-constants/`. It checks that the grid
  uuid of every forecast message matches the constants file.
- Station mapping: nearest neighbour on the unit sphere (as `station_grid_map.py`), cached in
  `resources/icon_station_map_ICON-CH{1,2}-EPS.csv`. CH1 cells equal the REA-L `grid_index` for
  all 145 stations (median distance 357 m); CH2 median 839 m, max 1683 m (2 km grid).
- Sanity check, init 2024-01-01 00 UTC vs REA-L at stations: T_2M r = 0.96 (CH2) / 0.99 (CH1);
  CH1 PS matches REA-L almost exactly (r = 1.000, RMSE 31 Pa), confirming how close REA-L is to
  ICON-CH1. CH2 PS is ~5 hPa lower on average (orography of the 2 km grid), irrelevant for
  band variance. Script: `output/icon_test/sanity.py`.
- Speed: 37 s per init for CH2 (121 steps), 15 s for CH1 (34 steps) on the login node, so SLURM
  is not needed. Full extraction started serially with `output/icon_test/run_all.sh`
  (log `output/icon_test/run_all.log`), output `output/points/<season>/ICON-CH{1,2}-CTRL/`.
- SLURM test job 5397150 (preemptible) was still queued; husk refused to cancel it because it was
  submitted in the previous session. Harmless. Louis may `scancel 5397150`.
- Scripts generalised (backups of the originals in the session scratchpad; git also has them):
  `spectra_forecast.load_season(..., extra=())` loads further sources and NaN-pads short ones
  (CH1) to 121 steps; `--extra-sources` / `--out-dir` added to `rmse_by_station_init.py`
  (+ `--max-step`, PS allowed vs REA-L), `bandpass_timedomain.py` (+ `--params`, PS gets FRAC
  only), `gap_robustness.py` (+ `--other`: model compared with Varda; fresh seed per run so the
  Multistep numbers stay reproducible), `joint_variance_skill.py` (extra models get
  frac_/r_/amp_/pha_ columns keyed e.g. `icon_ch2`); plotting: `skill_overview.py`
  (`--candidates`, `--params`, `--suffix`), `plot_frac_boxplots_overview.py` (n models, PS),
  `plot_paired_gap.py` (`--models LABEL:TAG`, one row each), `plot_timing_vs_variance.py`
  (`--dir`, `--models KEY:LABEL`). All take `--note` for a caveat line.
- Defaults-reproduce check: `output/icon_test/verify_defaults.sh` + `compare.py`
  (log `output/icon_test/verify.log`).
- Figure production: `output/icon_test/make_with_icon.sh` (log `output/icon_test/make_with_icon.log`).
  0-33 h skill uses `--min-steps 27` (80 % of 34, like 96 of 121 for the full window).
- **Defaults-reproduce check passed: 30 of 30 CSVs identical** (skill, rmse, bandpass order 4,
  gap_robustness incl. PS, joint_variance_skill incl. table.csv); all plot scripts ran.
- Extraction finished 21:34: all 164 ICON files (41 inits x 2 seasons x CH1/CH2), no NaN in
  T_2M/TD_2M/U/V/PS. Took ~2 h (about 45 s per init-pair on average, slower than the test).
- Figure production started 21:35.

### 2026-09-25 night

- Figures produced 21:35-21:45 (`output/icon_test/make_with_icon.log`, no errors). Adding
  ICON-CH2 left the matched sample unchanged (35 winter / 37 summer inits): all Multistep
  and Varda CSVs in `output/with_icon/` equal the existing ones.
- Summary numbers: `output/icon_test/summarise.py`.
- Plot fixes after seeing the numbers (see summary). Defaults rechecked: skill CSV identical,
  default timing and gap figures still render with the old layout.

## husk friction log (for the husk author), ordered by cost

1. **No passwd entry inside the sandbox** (`getent passwd $(id -u)` returns nothing). eckit's
   `~` expansion asserts on it (`Assertion failed: pw ... LocalPathName.cc expandTilde`) and
   kills the Python process on any ICON unstructured-grid load via earthkit. Probably a husk
   gap rather than policy: any library calling `getpwuid` breaks. Worked around by reading
   GRIB with eccodes directly. (Would also need `sites.ecmwf.int` for the grid download.)
2. **sbatch refused from a project under /users** ("Working directory ... is not allowed").
   Reasonable policy, but it cost a project migration to scratch. Only `preemptible` was
   allowed; `postproc` would have been the right partition.
3. **uv cache on scratch read-only**, PyPI blocked: `uv run --with ...` impossible; needed
   `uv pip install` outside husk and `UV_CACHE_DIR=<project>/.uvcache uv run --no-sync`.
4. **Home-based inputs invisible** (station-class tarball, ICON constants cache): copied into
   `resources/` by Louis. Expected policy.
5. **scancel refused for a job from the previous husk session** of the same user and project.
6. `sed -i` prints "preserving permissions ... Invalid argument" (the ACL/group mapping issue);
   edits still apply. Harmless but alarming.

### 2026-09-28: other time scales (Louis: do 6-12 h, daily cycle and synoptic, all with ICON)

- New `workflow/scripts/scales.py` defines four components of the 121 h record, all from the
  linearly detrended series: `3-6h` (original 4th-order bandpass, 24 h guards, bit-identical),
  `6-12h` (2nd-order bandpass, 26 h guards; 4th order would leave only 31 h), `diurnal`
  (least-squares 24 h + 12 h harmonics over lead 24-96 h; a bandpass kernel for 12-24 h is as
  long as the record), `synoptic` (24 h running mean applied twice, lead 24-96 h: zeroes 24 h
  and all harmonics, keeps ~2 days and longer). New scales are centred on their window
  because r uses mean(f x t) as covariance. Response check: `uv run workflow/scripts/scales.py`,
  plot `output/with_icon/scales/scales_response.png`.
- `--scale` added to gap_robustness, joint_variance_skill, bandpass_timedomain; label options
  to the plot scripts; `--from-table` to skill_overview (skill on the component, from the
  bmse columns of table.csv). **Defaults check rerun: 30/30 identical.**
- **Decision: every scale also in an "anomaly" variant** (`--anom`): each source's mean daily
  cycle (per station and season, by hour of day) removed first. Reason: the daily cycle has
  harmonics at 12, 8, 6, 4.8, 4 h that fall into the sub-daily bands and are sun-driven, so
  they inflate r "for free". Smoke test, winter T_2M, Varda r raw -> anom: 3-6h 0.19 -> 0.13,
  6-12h 0.78 -> 0.43, diurnal 0.96 -> 0.79, synoptic 0.92 -> 0.92.
- Production: `output/icon_test/make_scale.sh <scale> [anom]` -> `output/with_icon/scales/<scale>[_anom]/`
  (logs `output/icon_test/make_scale_<scale>.log`), then `plot_scales_summary.py [--anom]`.
- 2026-09-28: all 8 sets (4 scales x raw/anom) produced without errors, 107 figures, plus
  `output/with_icon/scales/scales_summary{,_anom}.png`. The raw 3-6h set equals the existing
  `output/with_icon/` results exactly. Numbers: `output/icon_test/summarise_scales.py`.
  Results written to FINDINGS.md section 15 (+ bottom-line point 8, notes in section 3).
  Headline: the variance deficit is sub-daily; timing improves steeply with scale; at no scale
  does multistep retain more or time better than Varda; multistep is less accurate on the
  synoptic component (-0.07 to -0.28 in 6 of 8 cases), which is where its full-series MSE
  deficit sits. Images not viewed (no viewer in husk).
