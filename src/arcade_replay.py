import os
import arcade
from src.interfaces.race_replay import F1RaceReplayWindow

# Kept these as "default" starting sizes, but they are no longer hard limits
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200
SCREEN_TITLE = "F1 Replay"

def run_arcade_replay(frames, track_statuses, example_lap, drivers, title,
                      playback_speed=1.0, driver_colors=None, circuit_rotation=0.0, total_laps=None, chart=False, win_prob_lookup=None):
    window = F1RaceReplayWindow(
        frames=frames,
        track_statuses=track_statuses,
        example_lap=example_lap,
        drivers=drivers,
        playback_speed=playback_speed,
        driver_colors=driver_colors,
        title=title,
        total_laps=total_laps,
        circuit_rotation=circuit_rotation,
        left_ui_margin=340,
        right_ui_margin=260,
        win_prob_lookup=win_prob_lookup,
    )
    arcade.run()
