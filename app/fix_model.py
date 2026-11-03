
import zipfile
from pathlib import Path

source = Path("../Models/best.pt")
fixed = Path("../Models/best_fixed.pt")

with zipfile.ZipFile(source, "r") as original:
    with zipfile.ZipFile(fixed, "w", zipfile.ZIP_DEFLATED) as repaired:
        for name in original.namelist():
            repaired.writestr("best/" + name, original.read(name))

print("Repaired archive created:", fixed.resolve())
