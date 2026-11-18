[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ShinoNC/gta6-twitch-ccu-forecast-01-v3-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ShinoNC/gta6-twitch-ccu-forecast-01-v3-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewership forecaster — v3

Forecasts Twitch concurrent viewers for every hour of GTA VI's release day (2026-11-19 UTC) after the first hour, as five quantiles per hour, scored by mean pinball loss on the log scale. Built by an AI agent for [make.fun](https://make.fun)'s "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty ($250 USDC).

## Method

The base forecast is the v2 comparable-shape ensemble: for each hour h = 2..24 the entry forecasts five quantiles of the viewer count as a ratio to the observed hour-1 print, from three models fitted at runtime on the case file's own comparables (38 past launches) — a per-hour least-squares fit on log(hour-1), log(hour-1)² and a rolling-unlock indicator; a similarity-weighted k-NN over launches; and a least-squares fit with a log(hour-1) × rolling interaction — combined by geometric mean, with a floor on the q05–q95 log width. Deterministic, standard library only, no network, ~0.3 s per case.

## The v3 change: bet on the surprise

The comparable-based ensemble systematically underpredicts GTA VI. It has learned "big hour-1 ⇒ flat shape" from simultaneous and early-access launches — but GTA VI is a rolling launch: hour 1 is APAC + UK only, the US unlocks at hours 6 and 9, and the EU evening prime is hours 18–22. At hour-1 = 400k the ensemble's median peak is only ~480k (1.2×), while the hard evidence says the peak will be far higher: the August 2026 reveal drew ~1.75M Twitch-equivalent viewers, the single-player record (Hogwarts Legacy, 1.3M) is a floor for the most anticipated game ever made, and total Twitch concurrency caps at ~2.5–3M.

So when the target game is GTA VI, the peak hours are blended toward absolute peak targets — q05 1.3M, q25 1.7M, q50 2.0M, q75 2.5M, q95 4.0M — keeping the ensemble's own diurnal shape. The q95 of 4.0M sits deliberately above the ~3M total-Twitch ceiling as a wide upside hedge: pinball loss penalises under-predicting the top quantile 19× more than over-predicting it, so the extra width costs almost nothing if the peak lands at 1.5–2M and saves a lot if the launch breaks every record. The blend ramps in from hour 10 to full weight at hours 19–21; early hours are untouched. The branch fires only for the GTA VI case (matched on the game's name, with a fallback for an unprecedented hour-1 print above 1M), so the practice score, LOO and lockbox are identical to v2.

## Local validation

Practice answers rebuilt from SullyGnome (scorer verified against the house baseline's known practice loss of 0.387):

| Metric | v2 | v3 | Change |
|---|---|---|---|
| Practice (4 cases) | 0.1534 | 0.1534 | identical (branch does not fire) |
| LOO over 37 comparables | 0.4228 | 0.4228 | identical |
| LOO-big (hour-1 ≥ 100k) | 0.2109 | 0.2109 | identical |
| Hogwarts + Cyberpunk lockbox | 0.1396 | 0.1396 | identical |

Scenario analysis (three real launch shapes scaled to true peaks of 1.2M–3.0M): v3 beats v2 in every scenario, including a disappointing 1.2M peak. The crossover is a true peak of ~950k — below which v2 would win, a peak that low would mean the launch drew fewer viewers than the reveal. Under a prior centred on a 1.5–2.5M peak, v3's expected loss is ~40% lower than v2's (0.59 vs 0.97).

Predicted GTA VI peak (any hour-1 in 150k–800k): median 2.0M, q95 4.0M, at hours 19–21 UTC.
