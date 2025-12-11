from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

@dataclass
class RaceMeta:
    season : int
    round_number : int
    event_name : str

def extract_race_meta(session) -> RaceMeta:
    event = session.event
    season = int(getattr(session, "year", event.get("EventDate").year))
    round_number = int(event.get("RoundNumber"))
    event_name = str(event.get("EventName"))
    return RaceMeta(season=season, round_number=round_number, event_name=event_name)

def build_race_state_from_session(session) -> pd.DataFrame:
    meta = extract_race_meta(session)
    laps = session.laps.copy()

    required = ["Driver", "LapNumber", "LapTime"]
    missing = [c for c in required if c not in laps.columns]
    if missing:
        raise KeyError(f"Laps is missing required columns: {missing}")

    if np.issubdtype(laps["LapTime"].dtype, np.timedelta64):
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    else:
        laps["LapTimeSeconds"] = pd.to_numeric(laps["LapTime"], errors="coerce")

    if "TyreLife" not in laps.columns:
        laps["TyreLife"] = np.nan

    if "Stint" not in laps.columns:
        laps["Stint"] = (
            laps.groupby("Driver")["Compound"].apply(lambda s: (s != s.shift()).cumsum())
            if "Compound" in laps.columns
            else 1
        )

    if "TrackStatus" not in laps.columns:
        laps["TrackStatus"] = 1

    if "IsPersonalBest" not in laps.columns:
        laps["IsPersonalBest"] = False

    if "PitInTime" in laps.columns:
        laps["PitIn"] = laps["PitInTime"].notna().astype(int)
    else:
        laps["PitIn"] = 0

    laps["Season"] = meta.season
    laps["RoundNumber"] = meta.round_number
    laps["EventName"] = meta.event_name

    laps["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce").fillna(0).astype(int)

    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    laps["RollingLapMean_3"] = (
        laps.groupby("Driver")["LapTimeSeconds"].transform(lambda s: s.rolling(window = 3, min_periods=1).mean())
    )
    laps["RollingLapMean_5"] = (
        laps.groupby("Driver")["LapTimeSeconds"].transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )

    field_mean = (
        laps.groupby("LapNumber")["LapTimeSeconds"].transform("mean")
    )

    laps["BestLapSoFar"] = (
        laps.groupby("Driver")["LapTimeSeconds"].transform("cummin")
    )
    laps["DeltaVSPersonalBest"] = laps["LapTimeSeconds"] - laps["BestLapSoFar"]

    laps["CumulTime"] = (
        laps.groupby("Driver")["LapTimeSeconds"].transform("cumsum")
    )

    leader_time = (
        laps.groupby("LapNumber")["CumulTime"].transform("min")
    )
    laps["GapToLeader"] = laps["CumulTime"] - leader_time

    laps["GapVolatility_3"] = (
        laps.groupby("Driver")["GapToLeader"].transform(lambda s: s.rolling(window = 3, min_periods=1).std())
    )

    laps["TotalLaps"] = laps["LapNumber"].max()
    laps["LapsRemaining"] = laps["TotalLaps"] - laps["LapNumber"]
    laps["RaceProgress"] = laps["LapNumber"] / laps["TotalLaps"]

    laps["StintLapIndex"] = (
        laps.groupby(["Driver", "Stint"]).cumcount() + 1
    )
    laps["NumPitstopsSoFar"] = laps.groupby("Driver")["PitIn"].cumsum()

    cols = [
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

    cols = [c for c in cols if c in laps.columns]
    race_state = laps[cols].copy()
    return race_state