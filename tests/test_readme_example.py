"""The first-user example must run with an installed package and keep its printed result."""
from pathlib import Path
import subprocess
import sys

def test_readme_example():
    root=Path(__file__).resolve().parents[1]
    completed=subprocess.run([sys.executable,str(root/"examples/quickstart.py")],
                             cwd=root,text=True,capture_output=True,check=True)
    expected=(root/"examples/expected_output.txt").read_text(encoding="utf-8")
    assert completed.stdout.strip()==expected.strip()
