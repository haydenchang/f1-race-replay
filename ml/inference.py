from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
from xgboost import XGBClassifier

from .feature_engineering import build_race_state_from_session

FEATURE_COLS = [
    "Season",
    "RoundNumber",
    "EventName",
    "Driver",
    "LapNumber",
    "LapTimeSeconds",
    "TyreLife",
    "Position",
    "Stint",
    "TrackStatus",
    "IsPersonalBest",
    "RollingLapMean_3",
    "RollingLapMean_5",
    "LapDeltaVSFieldMean",
    "BestLapSoFar",
    "DeltaVSPersonalBest",
    "CumulTime",
    "GapToLeader",
    "GapVolatility_3",
    "TotalLaps",
    "LapsRemaining",
    "RaceProgress",
    "StintLapIndex",
    "PitIn",
    "NumPitstopsSoFar",
]

def compute_win_probabilities_for_session(
    session,
    model : XGBClassifier,
) -> Dict[Tuple[str, int], float]:
    race_state = build_race_state_from_session(session)

    booster = model.get_booster()
    feature_names = booster.feature_names

    if feature_names is None:
        from .train_model import FEATURE_COLS
        feature_names = [c for c in FEATURE_COLS if c in race_state.columns]

    missing = [f for f in feature_names if f not in race_state.columns]
    if missing:
        for col in missing:
            race_state[col] = 0.0

    X = race_state[feature_names].copy()

    X = X.fillna(X.mean(numeric_only=True))
    probs = model.predict_proba(X)[:, 1]

    min_prob = 0.001
    max_prob = 0.999
    probs = np.clip(probs, min_prob, max_prob)
    win_prob_lookup = {}
    drivers = race_state["Driver"].astype(str).values
    laps = race_state["LapNumber"].astype(int).values

    for driver, lap, p in zip(drivers, laps, probs):
        key = (driver, int(lap))
        win_prob_lookup[key] = float(p)

    return win_prob_lookup