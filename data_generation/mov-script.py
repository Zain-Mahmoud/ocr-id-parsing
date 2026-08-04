from pathlib import Path
import random
import shutil

random.seed(42)  # Remove or change for different random samples

SOURCE = {
    "male": Path("/Users/zain/Downloads/faces/man"),
    "female": Path("/Users/zain/Downloads/faces/woman"),
}

DEST = Path("/Users/zain/EGID/images")

# Supported image extensions
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for gender, src_dir in SOURCE.items():
    dst_dir = DEST / gender
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = [
        f for f in src_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ]

    if len(images) < 100:
        raise ValueError(
            f"Only found {len(images)} images in {src_dir}, need at least 100."
        )

    selected = random.sample(images, 100)

    for img in selected:
        shutil.copy2(img, dst_dir / img.name)

    print(f"Copied {len(selected)} {gender} images.")

print("Done!")