import os
import subprocess

# URLs for each class
URLS = {
    "walking": [
        #"https://youtu.be/84lYjtCfIvY",
        #"https://youtu.be/3FXUw98rrUY",
        #"https://youtu.be/QBr2e3lnDsw",
        #"https://youtu.be/g0lMymp-FUc",
        #"https://youtu.be/eCNyPndCFQM",
        #"https://youtube.com/shorts/ZbMZtq2dNGw",
        #"https://youtu.be/ZvzKuqSDyG8",
        #"https://youtu.be/VamCnoHZezg",
    ],
    "running": [
        #"https://youtu.be/se1RDOPvA8Q",
        #"https://youtu.be/w_g1i6tzNGk",
        #"https://youtu.be/C1rmPz08SC0",
        #"https://youtu.be/GMrWISfQwBA",
        #"https://youtu.be/EA7mFlSuLhY",
        #"https://youtube.com/shorts/y9dq8Fk-0g4",
    ],
    "falling": [
    	"https://youtube.com/shorts/BGBlirNibIo?si=o7q8DMEE9A22APm6",
        #"https://youtube.com/shorts/6y83yGXgxdQ",
        #"https://youtube.com/shorts/3NQjUzzTxoA",
        #"https://youtu.be/ZSTflpwEPnw",
        #"https://youtube.com/shorts/D7c2TRmvrqY",
        #"https://youtube.com/shorts/ZqWnDgSujHY",
    ],
}

# Output dataset directory
DATASET_DIR = os.path.expanduser("~/activity_dataset")
CLIP_DURATION = 5  # seconds per clip
CLIP_OVERLAP = 2   # seconds of overlap between clips

def download_video(url, output_path):
    cmd = [
        "yt-dlp",
        "-f", "mp4/best[height<=480]",
        "--no-playlist",
        "-o", output_path,
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR downloading {url}: {result.stderr[:200]}")
        return False
    return True

def split_into_clips(video_path, output_dir, label, video_idx):
    # Get video duration
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        duration = float(result.stdout.strip())
    except:
        print(f"  Could not get duration for {video_path}")
        return 0

    clip_count = 0
    start = 0
    while start + CLIP_DURATION <= duration:
        clip_name = f"{label}_v{video_idx}_c{clip_count:03d}.mp4"
        clip_path = os.path.join(output_dir, clip_name)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(CLIP_DURATION),
            "-vf", "scale=182:182",
            "-r", "30",
            "-c:v", "libx264",
            "-an",  # no audio
            clip_path
        ]
        subprocess.run(cmd, capture_output=True)
        clip_count += 1
        start += (CLIP_DURATION - CLIP_OVERLAP)

    return clip_count

def main():
    # Create directories
    for split in ["train", "val"]:
        for label in URLS.keys():
            os.makedirs(os.path.join(DATASET_DIR, split, label), exist_ok=True)

    tmp_dir = os.path.expanduser("~/activity_dataset/tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for label, urls in URLS.items():
        print(f"\n{'='*40}")
        print(f"Processing class: {label} ({len(urls)} videos)")
        print(f"{'='*40}")

        all_clips = []

        for i, url in enumerate(urls):
            print(f"\n[{i+1}/{len(urls)}] Downloading: {url}")
            tmp_path = os.path.join(tmp_dir, f"{label}_{i}.mp4")

            if not download_video(url, tmp_path):
                continue

            print(f"  Splitting into {CLIP_DURATION}s clips...")
            # Put clips temporarily in train folder, we'll split later
            train_dir = os.path.join(DATASET_DIR, "train", label)
            count = split_into_clips(tmp_path, train_dir, label, i)
            print(f"  Generated {count} clips")
            all_clips.extend([f"{label}_v{i}_c{j:03d}.mp4" for j in range(count)])

            # Remove temp video
            os.remove(tmp_path)

        # Move 20% of clips to val
        print(f"\nSplitting {label} into train/val...")
        train_dir = os.path.join(DATASET_DIR, "train", label)
        val_dir = os.path.join(DATASET_DIR, "val", label)

        clips = sorted(os.listdir(train_dir))
        val_count = max(1, len(clips) // 5)
        val_clips = clips[-val_count:]

        for clip in val_clips:
            src = os.path.join(train_dir, clip)
            dst = os.path.join(val_dir, clip)
            os.rename(src, dst)

        print(f"  Train: {len(clips) - val_count} clips, Val: {val_count} clips")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*40}")
    print("Dataset preparation complete!")
    print(f"{'='*40}")
    for split in ["train", "val"]:
        for label in URLS.keys():
            path = os.path.join(DATASET_DIR, split, label)
            count = len(os.listdir(path))
            print(f"  {split}/{label}: {count} clips")

if __name__ == "__main__":
    main()
