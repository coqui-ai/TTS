import subprocess
import sys
from pathlib import Path


def test_readme_up_to_date():
    root = Path(__file__).parent.parent.parent
    sync_readme = root / "scripts" / "sync_readme.py"
    subprocess.check_call([sys.executable, str(sync_readme), "--check"], cwd=root)
