#
# AVI to MP4 Video Converter using Python's subprocess module and FFmpeg
#
# This script no longer uses the 'moviepy' library to resolve import issues.
# It now calls the external command-line tool 'FFmpeg' directly.
#
# DEPENDENCY:
# The 'FFmpeg' command-line utility MUST be installed on your system
# and accessible from your system's PATH.
#
import os
import sys
import subprocess
import shlex  # Used for safe command string splitting


def convert_avi_to_mp4(input_filepath, output_filepath=None):
    """
    Converts a video file from AVI format to MP4 format by executing the
    FFmpeg command-line tool via subprocess.

    Args:
        input_filepath (str): The full path to the input .avi file.
        output_filepath (str, optional): The full path for the output .mp4 file.
                                         If None, it defaults to the input filename
                                         with the .mp4 extension.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    # 1. Check if the input file exists
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at '{input_filepath}'")
        return False

    # 2. Determine the output file path
    if output_filepath is None:
        # Create default output path by replacing the extension with .mp4
        base_name, ext = os.path.splitext(input_filepath)
        # Ensure we only use .mp4 extension even if the input had a typo
        output_filepath = base_name + ".mp4"

    print(f"--- Starting conversion via FFmpeg subprocess ---")
    print(f"Input: {input_filepath}")
    print(f"Output: {output_filepath}")

    # 3. Construct the FFmpeg command
    # -i: Input file
    # -c:v libx264: Video codec (H.264 is standard for MP4)
    # -c:a aac: Audio codec (AAC is standard for MP4)
    # -y: Overwrite output file without asking
    # -loglevel error: Suppress excessive FFmpeg output, only show errors

    # We use shlex.quote for safety, especially if file paths contain spaces
    command_template = (
        "ffmpeg -i {input_path} -c:v libx264 -c:a aac -y -loglevel error {output_path}"
    )

    # Use shlex.quote to safely handle file paths with spaces
    ffmpeg_command = command_template.format(
        input_path=shlex.quote(input_filepath), output_path=shlex.quote(output_filepath)
    )

    # Split the command string into a list for subprocess
    command_list = shlex.split(ffmpeg_command)

    try:
        # 4. Execute the FFmpeg command
        result = subprocess.run(
            command_list,
            check=True,  # Raise an exception for non-zero return codes (errors)
            capture_output=True,  # Capture stdout and stderr
            text=True,
        )

        print(f"\n--- Conversion successful! ---")
        print(f"File saved to: {output_filepath}")
        return True

    except subprocess.CalledProcessError as e:
        # This occurs if FFmpeg returns a non-zero exit code (e.g., file corruption, codec issue)
        print(f"\n--- Conversion failed! (FFmpeg Error) ---")
        print(f"Command executed: {e.cmd}")
        print(f"FFmpeg Stderr:\n{e.stderr.strip()}")
        print("Possible causes: Input file corrupt, or a codec issue.")
        return False

    except FileNotFoundError:
        # This occurs if the 'ffmpeg' executable cannot be found
        print(f"\n--- Conversion failed! (FFmpeg Not Found) ---")
        print("Error: The 'ffmpeg' command was not found.")
        print("Please ensure FFmpeg is installed and added to your system's PATH.")
        return False

    except Exception as e:
        # Catch any other unexpected errors
        print(f"\n--- Conversion failed! (General Error) ---")
        print(f"An unexpected error occurred: {e}")
        return False


if __name__ == "__main__":
    # --- Example Usage ---

    # Default file names
    INPUT_FILE = "input_video.avi"  # <-- RENAME THIS to your actual AVI file
    OUTPUT_FILE = None  # Set to None to use default naming (e.g., input_video.mp4)
    # OUTPUT_FILE = "my_converted_video.mp4" # <-- Or specify a custom output name

    # Check if a filename was passed as a command-line argument
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
        if len(sys.argv) > 2:
            OUTPUT_FILE = sys.argv[2]
        else:
            OUTPUT_FILE = None

    print("----------------------------------------------------------------")
    print("Video Converter Initialized (Using FFmpeg directly).")
    print("Ensure the FFmpeg command-line tool is installed and in your PATH.")
    print("Run this script in the terminal like this:")
    print(f"  python {os.path.basename(__file__)} my_old_movie.avi my_new_movie.mp4")
    print("----------------------------------------------------------------")

    convert_avi_to_mp4(INPUT_FILE, OUTPUT_FILE)
