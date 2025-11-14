import sys
import subprocess
import os

print("--- Python Environment Diagnostics ---")

# 1. Check Python Version and Path
print(f"1. Current Python Interpreter Path: {sys.executable}")
print(f"2. Current Python Version: {sys.version.split()[0]}")

# 3. Check if 'moviepy' is importable
try:
    import moviepy.editor

    print("3. SUCCESS: 'moviepy' is importable.")
except ImportError:
    print("3. FAILURE: 'moviepy' is NOT importable in this environment.")
    print("   This confirms the problem is an environment/path mismatch.")

# 4. Check 'pip list' for installed packages (requires system call)
print("\n4. Running 'pip list' to check for installed 'moviepy':")
try:
    # Use the same executable path to ensure we check the correct environment's packages
    pip_path = os.path.join(os.path.dirname(sys.executable), "pip")

    # Try using 'pip' or 'pip3'
    result = subprocess.run(
        [pip_path, "list"], capture_output=True, text=True, check=True
    )

    found = False
    for line in result.stdout.splitlines():
        if "moviepy" in line.lower():
            print(f"   [FOUND] {line.strip()}")
            found = True

    if not found:
        print("   [NOT FOUND] 'moviepy' does not appear in the installed package list.")

except Exception as e:
    print(f"   Could not run 'pip list' check: {e}")


print("\n--- Next Steps ---")
print(
    "If step 3 fails, but step 4 finds 'moviepy', please ensure you are running your script using the exact Python executable shown in step 1."
)
