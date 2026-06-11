"""
Integrate a Label Studio YOLO export into yolo_dataset/.

Label Studio exports class IDs 0-based relative to its own project (0–67 for
the 68 new shoulder/orthopaedic classes).  This script offsets each ID by 26
so they align with the global 94-class list, then appends the images and labels
to yolo_dataset/images/train| val and yolo_dataset/labels/train|val.

Usage:
    python integrate_new_labels.py --export <label_studio_export_dir>

Expected export layout (standard Label Studio YOLO export):
    <export_dir>/
        images/   *.jpg / *.png
        labels/   *.txt   (one line per box: class x y w h)

The script splits 90 % → train, 10 % → val (reproducible, sorted order).
Re-running is safe: existing files are overwritten, nothing is deleted.
"""

import argparse
import random
import shutil
from pathlib import Path

# Number of OSI-26 classes that precede the new shoulder classes.
CLASS_ID_OFFSET = 26

YOLO_DIR = Path(__file__).parent / "yolo_dataset"
VAL_FRACTION = 0.10


def offset_label_file(src: Path, dst: Path, offset: int) -> None:
    lines = src.read_text().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        parts[0] = str(int(parts[0]) + offset)
        new_lines.append(" ".join(parts))
    dst.write_text("\n".join(new_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True,
                        help="Path to Label Studio YOLO export directory")
    args = parser.parse_args()

    export_dir = Path(args.export)
    img_src = export_dir / "images"
    lbl_src = export_dir / "labels"

    if not img_src.exists() or not lbl_src.exists():
        raise SystemExit(f"Expected {img_src} and {lbl_src} — check export path.")

    # Only process images that have a matching label file.
    label_files = sorted(lbl_src.glob("*.txt"))
    paired = []
    for lbl in label_files:
        for ext in (".jpg", ".jpeg", ".png"):
            img = img_src / (lbl.stem + ext)
            if img.exists():
                paired.append((img, lbl))
                break
        else:
            print(f"  [WARN] no image for {lbl.name} — skipping")

    if not paired:
        raise SystemExit("No matched image/label pairs found.")

    # Deterministic train/val split.
    random.seed(42)
    paired_shuffled = paired.copy()
    random.shuffle(paired_shuffled)
    n_val = max(1, int(len(paired_shuffled) * VAL_FRACTION))
    val_set  = set(p[0].name for p in paired_shuffled[:n_val])

    counts = {"train": 0, "val": 0}
    for img_path, lbl_path in paired:
        split = "val" if img_path.name in val_set else "train"

        out_img = YOLO_DIR / "images" / split / img_path.name
        out_lbl = YOLO_DIR / "labels" / split / (lbl_path.stem + ".txt")

        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, out_img)
        offset_label_file(lbl_path, out_lbl, CLASS_ID_OFFSET)
        counts[split] += 1

    print(f"Integrated {counts['train']} train  +  {counts['val']} val samples → {YOLO_DIR}")
    print("Next step: re-run prepare_dataset.py to regenerate dataset.yaml, then train.py")


if __name__ == "__main__":
    main()
