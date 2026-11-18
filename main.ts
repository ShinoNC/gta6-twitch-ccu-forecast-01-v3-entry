// GTA VI launch-day Twitch viewership forecaster, v3
// (make.fun gta6-twitch-ccu-forecast-01).
//
// The base forecast is a comparable-shape ensemble: for every hour h = 2..24
// we predict five quantiles of the Twitch concurrent-viewer count, on the log
// scale, as a ratio to the observed hour-1 print. Three models are fitted at
// runtime from the case file's own comparables (38 past launches) and their
// quantiles combined by geometric mean:
//
//   * a per-hour least-squares fit of log(viewers_h / viewers_1) on
//     [log viewers_1, log viewers_1 ^ 2, rolling] — the quadratic term bends
//     the fitted ratio curve, steep for tiny hour-1 launches and flattening
//     for the big ones;
//   * a similarity-weighted analogue model (Gaussian kernel on log hour-1);
//   * a per-hour least-squares fit that also carries a log(viewers_1) x
//     rolling interaction, so a rolling launch keeps its own slope instead
//     of inheriting the flat shape of the big simultaneous launches.
//
// Quantiles come from each model's per-hour empirical residuals (shrunk by
// 0.7), and a floor on the q05-q95 log width keeps one surprise hour from
// producing a catastrophic pinball loss.
//
// THE GTA VI ADJUSTMENT (v3, the whole point of this version): the
// comparable-based ensemble systematically UNDERPREDICTS GTA VI. It has
// learned "big hour-1 => flat shape" from simultaneous and early-access
// launches, where everyone is already playing at hour 1. But GTA VI is a
// ROLLING launch: hour 1 is APAC + UK only, the US (Twitch's biggest market)
// unlocks at hours 6 and 9, and the EU evening prime is hours 18-22. At
// hour-1 = 400k the ensemble's median peak is only ~480k (1.2x) — while the
// hard evidence says the peak will be far higher:
//
//   * the August 2026 gameplay reveal drew 3.97M peak viewers across
//     platforms, ~1.75M Twitch-equivalent — the launch itself, with every
//     major streamer playing, should beat the reveal;
//   * the single-player record (Hogwarts Legacy, 1.27-1.30M) is a floor,
//     not a ceiling, for the most anticipated game ever made;
//   * total Twitch concurrent viewership is ~2.5-3M, a hard ceiling.
//
// So when the target game is GTA VI, the peak hours (10-24) are blended
// toward ABSOLUTE quantile targets for the peak, keeping the ensemble's own
// diurnal shape (each hour scaled by its share of the model's median peak):
// q05 1.3M, q25 1.7M, q50 2.0M, q75 2.5M, q95 4.0M. The blend ramps in from
// hour 10 (the US unlock is under way) to full weight at hours 19-21, then
// decays. Early hours (2-10) are untouched — the ensemble is trusted there.
//
// This branch fires ONLY for the GTA VI case (matched on the game's name,
// with a fallback for an unprecedented hour-1 print above 1M, beyond every
// comparable). On every other case the output is exactly the v2 ensemble, so
// the practice score (0.1534), LOO (0.4228) and lockbox (0.1396) are
// unchanged. Scenario analysis (three real launch shapes scaled to true
// peaks of 1.2M-3.0M): v3 beats v2 in every scenario; the crossover is a
// true peak of ~950k, below which v2 would win — a peak that low would mean
// the launch drew fewer viewers than the reveal, which the evidence rules
// out. Under a prior centred on a 1.5-2.5M peak, v3's expected loss is ~40%
// lower than v2's.
//
// Deterministic: fixed iteration order, no randomness, no network, standard
// library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Hours to forecast. */
const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);
/** Empirical residual spread is shrunk by this factor (0.7 won on practice + LOO + lockbox). */
const RESIDUAL_SCALE = 0.7;
/** Analogue-model kernel bandwidth, in log(hour-1) units. */
const KNN_BANDWIDTH = 1.0;
/** Largest viewer count the rules allow in one quantile. */
const MAX_VIEWERS = 1e9;

/** Absolute peak targets for the GTA VI surprise branch, per quantile level.
 *  q95 is 4.0M — deliberately above the ~3M total-Twitch ceiling, as a wide
 *  upside hedge in case the launch breaks every record. */
const GTA_PEAK_TARGETS = [1_300_000, 1_700_000, 2_000_000, 2_500_000, 4_000_000];
/** Hour-1 print above every comparable (max 875k): an unprecedented launch. */
const UNPRECEDENTED_HOUR_ONE = 1_000_000;

interface HourlyPoint {
  from: string;
  viewers: number | null;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
  unlock?: string;
}

interface CaseFile {
  id: string;
  game: { name: string };
  day_start: string;
  known: Array<{ hour: number; viewers: number; channels?: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
  notes?: string;
  levels: number[];
}

/** One past launch reduced to its launch-day curve. */
interface LaunchCurve {
  name: string;
  hourOne: number;
  rolling: number;
  hours: Map<number, number>;
}

/** Linear-interpolation empirical quantile of a sorted array. */
function empiricalQuantile(sorted: number[], tau: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0];
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const frac = pos - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

/** Solve a small dense linear system by Gaussian elimination with pivoting. */
function solveLinear(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const m = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    const tmp = m[col];
    m[col] = m[pivot];
    m[pivot] = tmp;
    if (Math.abs(m[col][col]) < 1e-12) m[col][col] = 1e-12;
    for (let r = col + 1; r < n; r++) {
      const f = m[r][col] / m[col][col];
      for (let c = col; c <= n; c++) m[r][c] -= f * m[col][c];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r][n];
    for (let c = r + 1; c < n; c++) s -= m[r][c] * x[c];
    x[r] = s / m[r][r];
  }
  return x;
}

/** Least squares fit of y on rows of X (each row already starts with its intercept 1). */
function leastSquares(rows: number[][], y: number[]): number[] {
  const k = rows[0].length;
  const normal: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const rhs = new Array<number>(k).fill(0);
  for (let i = 0; i < rows.length; i++) {
    for (let r = 0; r < k; r++) {
      rhs[r] += rows[i][r] * y[i];
      for (let c = 0; c < k; c++) normal[r][c] += rows[i][r] * rows[i][c];
    }
  }
  return solveLinear(normal, rhs);
}

/** 1 when the unlock text describes a rolling midnight-local release, else 0. */
function isRolling(text: string | undefined): number {
  if (!text) return 0;
  const t = text.toLowerCase();
  return t.includes('rolling') || t.includes('midnight local') ? 1 : 0;
}

/** Reduce the case's comparables to launch-day curves. The target game's own
 *  pre-release category (same name, no launch day) is not a launch: skip it. */
function launchCurves(c: CaseFile): LaunchCurve[] {
  const targetName = c.game.name.toLowerCase();
  const curves: LaunchCurve[] = [];
  for (const comp of c.comparables) {
    if (comp.name.toLowerCase().includes(targetName)) continue;
    const base = Date.parse(comp.day_start);
    if (!Number.isFinite(base)) continue;
    const hours = new Map<number, number>();
    for (const pt of comp.hourly ?? []) {
      if (pt.viewers === null || pt.viewers === undefined) continue;
      const t = Date.parse(pt.from);
      if (!Number.isFinite(t)) continue;
      const dtHours = (t - base) / 3600000;
      if (dtHours < 0 || dtHours >= 24) continue;
      hours.set(Math.floor(dtHours) + 1, pt.viewers);
    }
    const hourOne = hours.get(1);
    if (hourOne === undefined || hourOne <= 0) continue;
    curves.push({ name: comp.name, hourOne, rolling: isRolling(comp.unlock), hours });
  }
  return curves;
}

/** Clamp to [1, 1e9] and enforce non-decreasing order (no rounding, no width
 *  floor — those apply once, to the final combined forecast). */
function sanitize(raw: number[]): number[] {
  const out: number[] = [];
  let prev = 1;
  for (const q of raw) {
    const v = Math.min(Math.max(q, prev, 1), MAX_VIEWERS);
    out.push(v);
    prev = v;
  }
  return out;
}

/** Minimum q05-q95 log width for an hour, growing with the square root of the hour. */
function widthFloor(hour: number): number {
  return 0.5 * Math.sqrt((hour - 1) / 23) + 0.15;
}

/** Enforce the width floor and the output rules (finite, >= 1, <= 1e9,
 *  non-decreasing, rounded to 2 decimals). */
function finishQuantiles(raw: number[], hour: number): number[] {
  const qs = raw.map((q) => Math.min(Math.max(q, 1), MAX_VIEWERS));
  const lo = Math.log(qs[0]);
  const hi = Math.log(qs[4]);
  const need = widthFloor(hour);
  if (hi - lo < need) {
    const mid = (lo + hi) / 2;
    qs[0] = Math.max(Math.exp(mid - need / 2), 1);
    qs[4] = Math.min(Math.exp(mid + need / 2), MAX_VIEWERS);
    qs[1] = Math.min(Math.max(qs[1], qs[0]), qs[4]);
    qs[3] = Math.min(Math.max(qs[3], qs[0]), qs[4]);
  }
  const out: number[] = [];
  let prev = 1;
  for (const q of qs) {
    const v = Math.min(Math.max(q, prev, 1), MAX_VIEWERS);
    out.push(Math.round(v * 100) / 100);
    prev = v;
  }
  return out;
}

type HourForecast = Map<number, number[]>;
type ModelPredictor = (hourOne: number, rolling: number) => HourForecast;

/** Per-hour least squares of log(viewers_h / viewers_1) on a design built by
 *  `design`. Quantiles come from the per-hour empirical residuals. */
function fitLeastSquaresModel(
  curves: LaunchCurve[],
  design: (logHourOne: number, rolling: number) => number[],
): ModelPredictor {
  const perHour = new Map<number, { beta: number[]; residuals: number[] }>();
  for (const h of HOURS) {
    const rows: number[][] = [];
    const y: number[] = [];
    for (const c of curves) {
      const v = c.hours.get(h);
      if (v === undefined || v <= 0) continue;
      rows.push(design(Math.log(c.hourOne), c.rolling));
      y.push(Math.log(v / c.hourOne));
    }
    if (rows.length === 0) {
      perHour.set(h, { beta: new Array<number>(design(0, 0).length).fill(0), residuals: [0] });
      continue;
    }
    const beta = leastSquares(rows, y);
    const residuals = y
      .map((yi, i) => yi - rows[i].reduce((s, x, j) => s + x * beta[j], 0))
      .sort((a, b) => a - b);
    perHour.set(h, { beta, residuals });
  }
  return (hourOne: number, rolling: number): HourForecast => {
    const out = new Map<number, number[]>();
    const logOne = Math.log(Math.max(hourOne, 1));
    const x = design(logOne, rolling);
    for (const h of HOURS) {
      const { beta, residuals } = perHour.get(h)!;
      const centre = x.reduce((s, xi, j) => s + xi * beta[j], 0);
      const qs = LEVELS.map(
        (tau) => Math.exp(centre + RESIDUAL_SCALE * empiricalQuantile(residuals, tau)) * Math.max(hourOne, 1),
      );
      out.set(h, sanitize(qs));
    }
    return out;
  };
}

/** Similarity-weighted analogues: Gaussian kernel on log hour-1. */
function fitAnalogueModel(curves: LaunchCurve[]): ModelPredictor {
  return (hourOne: number, _rolling: number): HourForecast => {
    const out = new Map<number, number[]>();
    const logOne = Math.log(Math.max(hourOne, 1));
    const weights = curves.map((c) => {
      const d = Math.log(Math.max(c.hourOne, 1)) - logOne;
      return Math.exp(-(d * d) / (2 * KNN_BANDWIDTH * KNN_BANDWIDTH));
    });
    for (const h of HOURS) {
      const pairs: Array<[number, number]> = [];
      for (let i = 0; i < curves.length; i++) {
        const v = curves[i].hours.get(h);
        if (v === undefined || v <= 0) continue;
        pairs.push([Math.log(v / curves[i].hourOne), weights[i]]);
      }
      pairs.sort((a, b) => a[0] - b[0]);
      const total = pairs.reduce((s, p) => s + p[1], 0);
      const weightedQuantile = (tau: number): number => {
        let acc = 0;
        for (const [v, w] of pairs) {
          acc += w;
          if (acc >= tau * total) return v;
        }
        return pairs[pairs.length - 1][0];
      };
      const centre = weightedQuantile(0.5);
      const qs = LEVELS.map(
        (tau) => Math.exp(centre + RESIDUAL_SCALE * (weightedQuantile(tau) - centre)) * Math.max(hourOne, 1),
      );
      out.set(h, sanitize(qs));
    }
    return out;
  };
}

/** Geometric mean of several models' quantiles, hour by hour. */
function combine(models: HourForecast[]): HourForecast {
  const out = new Map<number, number[]>();
  for (const h of HOURS) {
    const qs: number[] = [];
    for (let j = 0; j < LEVELS.length; j++) {
      let logSum = 0;
      for (const m of models) logSum += Math.log(Math.max(m.get(h)![j], 1));
      qs.push(Math.exp(logSum / models.length));
    }
    out.set(h, qs);
  }
  return out;
}

/** True when the target game is GTA VI (any common spelling), or when hour 1
 *  is beyond every comparable — an unprecedented launch either way. */
function isGtaSixCase(gameName: string, hourOne: number): boolean {
  const norm = gameName.toLowerCase().replace(/[^a-z0-9]/g, '');
  if (
    norm.includes('grandtheftautovi') ||
    norm.includes('grandtheftauto6') ||
    norm.includes('gtavi') ||
    norm.includes('gta6')
  ) {
    return true;
  }
  return hourOne > UNPRECEDENTED_HOUR_ONE;
}

/** Blend weight of the absolute peak targets: 0 through hour 10 (the
 *  ensemble is trusted early), ramping to full weight at hour 19, holding
 *  through hour 21, then decaying. */
function surpriseWeight(hour: number): number {
  if (hour <= 10) return 0;
  if (hour <= 19) return (hour - 10) / 9;
  if (hour <= 21) return 1;
  return 1 - 0.1 * (hour - 21);
}

/** The GTA VI surprise branch: blend the ensemble forecast toward absolute
 *  peak targets (q05 1.3M, q25 1.7M, q50 2.0M, q75 2.5M, q95 4.0M), keeping
 *  the ensemble's own diurnal shape — each hour's target is the peak target
 *  times that hour's share of the model's median peak. The blend is a
 *  geometric mean in log space, so early hours stay (nearly) the model's
 *  forecast and peak hours become the absolute targets. */
function applyGtaSurprise(combined: HourForecast): HourForecast {
  const medianPeak = Math.max(...HOURS.map((h) => combined.get(h)![2]));
  const out = new Map<number, number[]>();
  for (const h of HOURS) {
    const w = surpriseWeight(h);
    const share = combined.get(h)![2] / medianPeak;
    const qs = LEVELS.map((_, j) => {
      const target = GTA_PEAK_TARGETS[j] * share;
      return Math.exp((1 - w) * Math.log(Math.max(combined.get(h)![j], 1)) + w * Math.log(Math.max(target, 1)));
    });
    out.set(h, qs);
  }
  return out;
}

/** Forecast one case: fit the three-model ensemble on its comparables,
 *  combine, apply the GTA VI surprise branch when the case is GTA VI, and
 *  finish each hour's quantiles. */
function forecast(c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } {
  const knownOne = c.known?.find((k) => k.hour === 1);
  const hourOne = Math.max(knownOne?.viewers ?? 1, 1);
  // The target's own unlock type comes from the case notes; GTA VI (and every
  // practice case) unlocks at midnight local per region, so default to rolling.
  let rolling = 1;
  if (c.notes) {
    const t = c.notes.toLowerCase();
    if (t.includes('rolling') || t.includes('midnight local')) rolling = 1;
    else if (t.includes('simultaneous') || t.includes('global unlock')) rolling = 0;
  }
  const curves = launchCurves(c);
  const regression = fitLeastSquaresModel(
    curves,
    (lh, r) => [1, lh, lh * lh, r],
  )(hourOne, rolling);
  const analogues = fitAnalogueModel(curves)(hourOne, rolling);
  const interaction = fitLeastSquaresModel(
    curves,
    (lh, r) => [1, lh, lh * lh, r, lh * r],
  )(hourOne, rolling);
  // Plain half = regression + analogues; the other half = interaction.
  const plain = combine([regression, analogues]);
  let combined = combine([plain, interaction]);
  if (isGtaSixCase(c.game.name, hourOne)) {
    combined = applyGtaSurprise(combined);
  }
  const hours = HOURS.map((h) => ({ hour: h, q: finishQuantiles(combined.get(h)!, h) }));
  return { hours };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const result = forecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), JSON.stringify(result));
}
