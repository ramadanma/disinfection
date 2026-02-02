# motiondetector/services/maintenance.py
from pathlib import Path
from disinfection.core.logging_setup import cleanup_old_logs


def main():
    project_root = Path(__file__).resolve().parents[2]
    cleanup_old_logs(base_dir=str(project_root / "logs"), keep_days=30)


if __name__ == "__main__":
    main()
