import os
from pathlib import Path
from PIL import Image
import app

print("Checking GUI assets in Recibot/setUp...")
setup_dir = Path("Recibot/setUp")

expected_assets = [
    "yellow_bin.png", "yellow_bintxt.png",
    "organic.png", "organictxt.png",
    "paper.png", "papertxt.png",
    "glass.png", "glasstxt.png",
    "residual.png", "residualtxt.png",
    "deposit.png", "deposittxt.png",
    "Canva.png"
]

for asset in expected_assets:
    path = setup_dir / asset
    assert path.exists(), f"Missing expected asset: {asset}"
    img = Image.open(path)
    print(f" [OK] {asset:18s} size={img.size}")

print("\nChecking app.py rules and image groups...")
for key, rule in app.GERMAN_WASTE_RULES.items():
    assert "name" not in rule, f"Rule {key} still has 'name' field"
    assert "bin" in rule
    assert "group" in rule
    assert rule["group"] in ["yellow_bin", "organic", "paper", "glass", "residual", "deposit"], f"Invalid group in {key}: {rule['group']}"

print("ALL ASSET AND RULE CHECKS PASSED!")

