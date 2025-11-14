import os
import subprocess
import sys


def check_ffmpeg_installed():
    """Checks if FFmpeg is installed and accessible in the system's PATH."""
    try:
        # Run a simple command to check the version
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Error: FFmpeg failed to run. It may not be correctly installed.")
        return False
    except FileNotFoundError:
        print("\n❌ Error: The 'ffmpeg' command was not found.")
        print("Please ensure FFmpeg is installed and added to your system's PATH.")
        print(
            "Since you installed 'moviepy', FFmpeg *should* be installed, but it might not be in the system PATH."
        )
        return False


def convert_mp4_to_avi(input_file_path, output_file_path):
    """
    Converts an MP4 video file to an AVI file using the FFmpeg command line utility,
    preserving the original resolution.

    Args:
        input_file_path (str): The full path to the input .mp4 file.
        output_file_path (str): The full path where the output .avi file will be saved.
    """
    if not check_ffmpeg_installed():
        return

    # 1. Check if the input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    print(f"Starting conversion of: {input_file_path}")

    # FFmpeg command structure:
    # ffmpeg -i <input> -c:v <video_codec> -q:v 0 -c:a <audio_codec> <output>
    # -i: Input file
    # -c:v mpeg4: Set video codec to MPEG-4 (commonly used for AVI)
    # -q:v 0: Use the highest quality possible for the selected codec
    # -c:a pcm_s16le: Use uncompressed audio (good for AVI compatibility)
    # The resolution is kept by default unless a -s (size) parameter is specified.

    ffmpeg_command = [
        "ffmpeg",
        "-i",
        input_file_path,
        "-c:v",
        "mpeg4",
        "-q:v",
        "0",  # Highest quality
        "-c:a",
        "pcm_s16le",
        "-y",  # Overwrite output file without asking
        output_file_path,
    ]

    try:
        process = subprocess.run(
            ffmpeg_command,
            check=True,  # Raise CalledProcessError if return code is non-zero
            stdout=sys.stdout,
            stderr=sys.stderr,  # Print FFmpeg output directly to console
        )

        print("-" * 50)
        print(f"✅ Success! Video converted and saved to: {output_file_path}")
        print("-" * 50)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ FFmpeg command failed with error code {e.returncode}.")
        print("Check the output above for specific FFmpeg error messages.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


# --- Example Usage ---
# NOTE: Replace 'example_video.mp4' with the actual name and path of your file.
INPUT_FILE = "300VW_Dataset_2015_12_14/569/vid.mp4"
OUTPUT_FILE = "300VW_Dataset_2015_12_14/569/vid.avi"

if __name__ == "__main__":
    convert_mp4_to_avi(INPUT_FILE, OUTPUT_FILE)
