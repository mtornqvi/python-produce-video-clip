import os
from pathlib import Path

from dotenv import load_dotenv
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
import numpy as np


def load_settings(env_path: Path | str = ".env") -> dict[str, str]:
    env_path = Path(env_path)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    load_dotenv(dotenv_path=env_path)

    settings = {
        "jpg_source_folder": os.getenv("jpg_source_folder", "").strip(),
        "mp3_source_file": os.getenv("mp3_source_file", "").strip(),
        "mp4_target_folder": os.getenv("mp4_target_folder", "").strip(),
    }

    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return settings


def collect_image_files(source_folder: Path) -> list[Path]:
    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"JPEG source folder not found: {source_folder}")

    images = sorted(
        [
            path
            for path in source_folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ],
        key=lambda path: path.name,
    )

    if not images:
        raise ValueError(f"No JPEG images found in folder: {source_folder}")

    return images


def ensure_output_folder(target_folder: Path) -> None:
    target_folder.mkdir(parents=True, exist_ok=True)


def determine_frame_size(image_files: list[Path]) -> tuple[int, int]:
    max_width = 0
    max_height = 0

    print("Determining target frame size...")
    for image_path in image_files:
        with Image.open(image_path) as img:
            width, height = img.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)

    print(f"Target frame size: {max_width}x{max_height}")
    return max_width, max_height


def load_frame(image_path: Path, target_size: tuple[int, int], index: int, total: int) -> np.ndarray:
    if index % 20 == 0 or index == total:
        print(f"Loading frame {index}/{total}: {image_path.name}")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail(target_size, Image.LANCZOS)

        frame = Image.new("RGB", target_size, color="black")
        x = (target_size[0] - img.width) // 2
        y = (target_size[1] - img.height) // 2
        frame.paste(img, (x, y))

        return np.asarray(frame)


def build_video(
    image_files: list[Path],
    audio_file: Path,
    output_file: Path,
    fps: int = 30,
) -> None:
    target_size = determine_frame_size(image_files)
    total_frames = len(image_files)
    frames = [
        load_frame(path, target_size, index + 1, total_frames)
        for index, path in enumerate(image_files)
    ]

    print(f"Loaded {len(frames)} frames. Creating video clip...")
    clip = ImageSequenceClip(frames, fps=fps)
    audio_clip = AudioFileClip(str(audio_file))

    print(f"Audio duration: {audio_clip.duration:.2f}s, video duration: {clip.duration:.2f}s")
    if audio_clip.duration > clip.duration:
        print("Trimming audio to match video duration...")
        audio_clip = audio_clip.subclip(0, clip.duration)

    final_clip = clip.with_audio(audio_clip)
    print(f"Writing output file: {output_file}")
    final_clip.write_videofile(
        str(output_file),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(output_file.with_suffix(".temp.m4a")),
        remove_temp=True,
        write_logfile=False,
        logger=None,
    )


def main() -> None:
    settings = load_settings()
    jpg_source_folder = Path(settings["jpg_source_folder"])
    mp3_source_file = Path(settings["mp3_source_file"])
    mp4_target_folder = Path(settings["mp4_target_folder"])

    print("Loaded settings from .env:")
    print(f"  jpg_source_folder={jpg_source_folder}")
    print(f"  mp3_source_file={mp3_source_file}")
    print(f"  mp4_target_folder={mp4_target_folder}")

    image_files = collect_image_files(jpg_source_folder)
    if not mp3_source_file.exists():
        raise FileNotFoundError(f"Audio source file not found: {mp3_source_file}")

    ensure_output_folder(mp4_target_folder)
    output_file = mp4_target_folder / "output.mp4"

    print(f"Creating video with {len(image_files)} frames at 30 fps...")
    build_video(image_files, mp3_source_file, output_file)
    print(f"Video saved to: {output_file}")


if __name__ == "__main__":
    main()
