from roboflow import Roboflow
from pathlib import Path
import shutil
import yaml
from dotenv import load_dotenv
import os
from tqdm.auto import tqdm

load_dotenv()
API_KEY = os.environ['ROBOFLOW_TOKEN']

DATASETS = [
    ("idcard-b708e", "card_seg-4mofm", 1),
    ("nile-university-oions", "id-frame", 1),
    ("mo-1q5np", "card_finder-ykuhs", 2),
    ("alis-workspace-bjnpi", "detect-egyptian-national-id-jav3l-allbq", 1),
    ("roboflow-oa8fu", "grad-rvfr6", 2)
]

DOWNLOAD_DIR = Path("./downloads")

try:
    DOWNLOAD_DIR.mkdir()
except:
    shutil.rmtree("./downloads")
    DOWNLOAD_DIR.mkdir()

MERGED_DIR = Path("./data/detection-data")

try:
    MERGED_DIR.mkdir()
except:
    shutil.rmtree("./data/detection-data")
    MERGED_DIR.mkdir()

rf = Roboflow(api_key=API_KEY)

downloaded = []
os.chdir("./downloads")
for workspace, project, version in tqdm(DATASETS):

    ds = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download("yolo26")
    )

    downloaded.append(Path(ds.location))
os.chdir("..")

for split in ["train", "valid", "test"]:
    (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

counter = 0

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def polygon_to_bbox(coords):

    xs = coords[0::2]
    ys = coords[1::2]

    xmin = min(xs)
    xmax = max(xs)

    ymin = min(ys)
    ymax = max(ys)

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2

    w = xmax - xmin
    h = ymax - ymin

    return xc, yc, w, h


for dataset in tqdm(downloaded):

    for split in ["train", "valid", "test"]:

        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"

        if not image_dir.exists():
            continue

        for image in tqdm(image_dir.iterdir()):

            if image.suffix.lower() not in IMAGE_EXTS:
                continue

            new_name = f"{counter:08d}"

            shutil.copy2(
                image,
                MERGED_DIR / split / "images" / (new_name + image.suffix)
            )

            label = label_dir / (image.stem + ".txt")

            output_label = MERGED_DIR / split / "labels" / (new_name + ".txt")

            if label.exists():

                converted = []

                with open(label) as f:

                    for line in f:

                        vals = line.strip().split()

                        if len(vals) < 5:
                            continue

                        nums = list(map(float, vals[1:]))

                        # Detection:
                        # class xc yc w h
                        if len(nums) == 4:

                            xc, yc, w, h = nums

                        # Segmentation:
                        # class x1 y1 x2 y2 ...
                        else:

                            xc, yc, w, h = polygon_to_bbox(nums)

                        converted.append(
                            f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                        )

                with open(output_label, "w") as f:
                    f.write("\n".join(converted))

            counter += 1


yaml_data = {
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["id_card"],
}

with open(MERGED_DIR / "data.yaml", "w") as f:
    yaml.safe_dump(yaml_data, f, sort_keys=False)
