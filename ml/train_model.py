from __future__ import annotations
from pathlib import Path
from typing import List

import fastf1
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
from ml.feature_engineering import build_race_state_from_session
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "Season",
    "RoundNumber",
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

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "xgb_win_model.json"

def add_win_label(race_state: pd.DataFrame, session) -> pd.DataFrame:
    results = session.results
    winner_codes = results.loc[results["Position"] == 1, "Abbreviation"].astype(str).tolist()
    race_state["Driver"] = race_state["Driver"].astype(str)
    race_state["Win"] = race_state["Driver"].isin(winner_codes).astype(int)
    return race_state

def collect_training_data(
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    all_race_states: List[pd.DataFrame] = []
    fastf1.Cache.enable_cache(str(PROJECT_DIR / "fastf1_cache"))

    for year in range(start_season, end_season + 1):
        print(f"=== Season {year} ===")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Could not load schedule for {year}: {e}")
            continue

        for _, ev in schedule.iterrows():
            round_number = int(ev["RoundNumber"])
            event_name = str(ev["EventName"])

            if ev.get("EventFormat", "conventional") not in ["conventional", "sprint"]:
                continue

            print(f"  -> Loading race: Round {round_number} - {event_name}")

            try:
                session = fastf1.get_session(year, round_number, "R")
                session.load()
            except Exception as e:
                print(f"   ! Failed to load race session for {year} Round {round_number}: {e}")
                continue

            try:
                race_state = build_race_state_from_session(session)
                race_state = add_win_label(race_state, session)
                race_state = race_state.dropna(subset=["Win"])
                all_race_states.append(race_state)
                print(f"    collected {len(race_state)} rows")
            except Exception as e:
                print(f"    ! Failed to build race_state for {year} Round {round_number}: {e}")
                continue
    full_df = pd.concat(all_race_states, ignore_index=True)
    print(f"\nTotal training rows: {len(full_df)}")
    print("Class balance (Win):")
    print(full_df["Win"].value_counts())

    return full_df

def train_xgb():
    END = 2024
    START = 2024

    train_df = collect_training_data(START, END)

    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    X = train_df[available_features].copy()
    y = train_df["Win"].astype(int)

    X = X.fillna(X.mean(numeric_only=True))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print("\nTrain shape:", X_train.shape, "Val shape:", X_val.shape)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    xgb_clf = XGBClassifier(
        objective = "binary:logistic",
        eval_metric = "logloss",
        n_estimators = 500,
        learning_rate = 0.05,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        reg_lambda = 1.0,
        random_state = 42,
        n_jobs = -1,
        scale_pos_weight = scale_pos_weight,
        tree_method = 'hist',
    )

    xgb_clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], verbose = 50,)
    train_probs = xgb_clf.predict_proba(X_train)[:, 1]
    print("\nTrain probs stats: min =", train_probs.min(), "max =", train_probs.max())
    print("First 10 train probs:", train_probs[:10])

    val_probs = xgb_clf.predict_proba(X_val)[:, 1]
    print("Val probs stats: min =", val_probs.min(), "max =", val_probs.max())
    print("First 10 val probs:", val_probs[:10])

    val_probs = xgb_clf.predict_proba(X_val)[:, 1]
    print("\nValidation log loss:", log_loss(y_val, val_probs))
    print("Validation Brier:", brier_score_loss(y_val, val_probs))

    if not hasattr(xgb_clf, "_estimator_type"):
        xgb_clf._estimator_type = "classifier"

    xgb_clf.save_model(str(MODEL_PATH))
    print(f"\nSaved trained model to: {MODEL_PATH}")

    xgb_clf.save_model(str(MODEL_PATH))

if __name__ == "__main__":
    train_xgb()