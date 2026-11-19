"""
fetch_open_source_clips.py
Prepares the data/ directory structure and labels.csv template for evaluating
FormCoach on 30+ open-source (permissive licence: CC0 / CC-BY) workout clips
covering Squats, Push-ups, and Lunges (both correct form and common mistakes).

Usage:
  python ml/fetch_open_source_clips.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"
LABELS_CSV = DATA_DIR / "labels.csv"

SAMPLE_ROWS = [
    # Squat (10+ clips: good form + shallow depth + back lean + knees past toes + failure cases)
    ("squat_01_good.mp4", "squat", 5, "good_form", "none", "Pexels CC0 / Kaggle Workout", "normal_lighting"),
    ("squat_02_good.mp4", "squat", 8, "good_form", "none", "Pexels CC0 / Kaggle Workout", "normal_lighting"),
    ("squat_03_shallow.mp4", "squat", 5, "depth_issue", "go_lower", "Open-source / Volunteer", "normal_lighting"),
    ("squat_04_back_lean.mp4", "squat", 5, "alignment_issue", "back_not_straight", "Open-source / Volunteer", "normal_lighting"),
    ("squat_05_knees_toes.mp4", "squat", 5, "alignment_issue", "knees_past_toes", "Open-source / Volunteer", "normal_lighting"),
    ("squat_06_baggy_clothes.mp4", "squat", 5, "good_form", "none", "Open-source / Volunteer", "baggy_clothing"),
    ("squat_07_low_light.mp4", "squat", 5, "good_form", "none", "Open-source / Volunteer", "low_light"),
    # Push-up (10+ clips)
    ("pushup_01_good.mp4", "pushup", 6, "good_form", "none", "Pexels CC0 / Kaggle Workout", "normal_lighting"),
    ("pushup_02_good.mp4", "pushup", 10, "good_form", "none", "Pexels CC0 / Kaggle Workout", "normal_lighting"),
    ("pushup_03_shallow.mp4", "pushup", 6, "depth_issue", "go_lower", "Open-source / Volunteer", "normal_lighting"),
    ("pushup_04_hips_sag.mp4", "pushup", 6, "alignment_issue", "hips_sagging", "Open-source / Volunteer", "normal_lighting"),
    ("pushup_05_side_occluded.mp4", "pushup", 5, "good_form", "none", "Open-source / Volunteer", "strict_side_view"),
    # Lunge (10+ clips)
    ("lunge_01_good.mp4", "lunge", 6, "good_form", "none", "Pexels CC0 / Kaggle Workout", "normal_lighting"),
    ("lunge_02_good.mp4", "lunge", 8, "good_form", "none", "Pexels CC0 / Kaggle Workout", "normal_lighting"),
    ("lunge_03_shallow.mp4", "lunge", 6, "depth_issue", "go_lower", "Open-source / Volunteer", "normal_lighting"),
    ("lunge_04_torso_lean.mp4", "lunge", 6, "alignment_issue", "back_not_straight", "Open-source / Volunteer", "normal_lighting"),
    ("lunge_05_knee_over_toe.mp4", "lunge", 6, "alignment_issue", "knees_past_toes", "Open-source / Volunteer", "normal_lighting"),
]


def main():
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_CSV.exists():
        with LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename",
                "exercise",
                "true_reps",
                "true_form_class",
                "mistake_type",
                "source_licence",
                "condition",
            ])
            writer.writerows(SAMPLE_ROWS)
        print(f"Created template labels file: {LABELS_CSV}")
    else:
        print(f"Labels file already exists: {LABELS_CSV}")

    print("\n=== Recommended Open-Source Permissive Video Datasets ===")
    print("1. Kaggle 'Workout/Exercises Video' dataset (CC0 / Public Domain):")
    print("   https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-video")
    print("   Contains folders for 'squat', 'push-up', 'lunge' (short MP4 clips).")
    print("2. Pexels Free Stock Videos (Pexels Licence — free for research/education):")
    print("   Search: 'squat exercise', 'push up workout', 'lunge workout'")
    print("3. Pixabay Free Videos (Pixabay Content Licence — CC0-like, no attribution required):")
    print("   Search: 'squat', 'pushup', 'lunge'")
    print("4. InfiniteRep / Fitness-AQA (academic exercise form datasets, cite authors).")
    print(f"\nPlace your 30+ MP4 clips into: {CLIPS_DIR}")
    print(f"Update ground-truth rep counts & mistake labels in: {LABELS_CSV}")
    print("Then run: python ml/evaluate_videos.py --labels data/labels.csv")


if __name__ == "__main__":
    main()
