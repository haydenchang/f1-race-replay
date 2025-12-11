import argparse
from pathlib import Path

import fastf1
from xgboost import XGBClassifier

from src.f1_data import load_session, get_race_telemetry, get_driver_colors, get_circuit_rotation
from src.arcade_replay import run_arcade_replay
from ml.inference import compute_win_probabilities_for_session

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'xgb_win_model.json'


def load_trained_model(model_path: Path) -> XGBClassifier:
    model = XGBClassifier()
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    model.load_model(str(model_path))
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Run F1 Race Replay with Win Probabilities")
    parser.add_argument("--year", "-y", type = int, required = True, help = "Race year")
    parser.add_argument("--round", "-r", type = int, required = True, help = "Round number")
    return parser.parse_args()

def main():
    args = parse_args()
    YEAR = args.year
    ROUND = args.round

    cache_dir = PROJECT_DIR / 'fastf1_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    print(f"Loading race year = {YEAR}, round = {ROUND} ...")
    fastf1.Cache.enable_cache(str(PROJECT_DIR / "fastf1_cache"))

    session = load_session(YEAR, ROUND, session_type ="R")

    model = load_trained_model(MODEL_PATH)

    print("Computing win probabilities...")
    win_prob_lookup = compute_win_probabilities_for_session(session, model)
    print(f"win_prob: {len(win_prob_lookup)}")
    for i, (k, v) in enumerate(win_prob_lookup.items()):
        if i >= 5:
            break
        print(" sample:", k, "->", v)
    race_telemetry = get_race_telemetry(session, session_type ="R")
    example_lap = session.laps.pick_fastest().get_telemetry()
    drivers = session.drivers
    driver_colors = get_driver_colors(session)
    total_laps = int(session.laps["LapNumber"].max())
    circuit_rotation = get_circuit_rotation(session)

    event_name = session.event["EventName"]
    title = f"{event_name} {YEAR} - Win Probability Replay"
    run_arcade_replay(
        frames = race_telemetry["frames"],
        track_statuses=race_telemetry["track_statuses"],
        example_lap=example_lap,
        drivers=drivers,
        title=title,
        playback_speed=1.0,
        driver_colors=driver_colors,
        circuit_rotation=circuit_rotation,
        total_laps=total_laps,
        win_prob_lookup=win_prob_lookup,
    )

if __name__ == "__main__":
    main()