"""
Convert dataset1-pic-xml (Pascal VOC XML) → YOLO format.
Outputs:
  yolo_dataset/
    images/train/  images/val/
    labels/train/  labels/val/
  dataset.yaml
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ── OSI-26 class ID → human-readable name ─────────────────────────────────────
# Classes 1-26 are standard surgical instrument IDs from the OSI-26 benchmark.
CLASS_NAMES = {
    1:  "Needle Holder",
    2:  "Dissecting Forceps",
    3:  "Tissue Forceps",
    4:  "Mosquito Clamp",
    5:  "Kocher Clamp",
    6:  "Allis Clamp",
    7:  "Sponge Forceps",
    8:  "Babcock Clamp",
    9:  "Right Angle Clamp",
    10: "Towel Clamp",
    11: "Foerster Clamp",
    12: "Retractor",
    13: "Volkmann Curette",
    14: "Bone Curette",
    15: "Periosteal Elevator",
    16: "Osteotome",
    17: "Bone Chisel",
    18: "Mallet",
    19: "Bone Rasp",
    20: "Gigli Saw",
    21: "Wire Cutter",
    22: "Suture Scissors",
    23: "Mayo Scissors",
    24: "Metzenbaum Scissors",
    25: "Bandage Scissors",
    26: "Scalpel",
}

BASE      = Path(__file__).parent / "dataset1-pic-xml"
OUT_DIR   = Path(__file__).parent / "yolo_dataset"

SPLITS = {
    "train": ("train", "train"),
    "val":   ("test",  "val"),   # use test split as validation
}


def convert_xml_to_yolo(xml_path: Path, img_path: Path, label_out: Path, class_names: dict):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls_raw = int(obj.find("name").text)           # 1-based integer
        cls_idx = cls_raw - 1                          # 0-based YOLO index

        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        x_c = ((xmin + xmax) / 2) / W
        y_c = ((ymin + ymax) / 2) / H
        w   = (xmax - xmin) / W
        h   = (ymax - ymin) / H

        x_c, y_c, w, h = [round(v, 6) for v in (x_c, y_c, w, h)]
        lines.append(f"{cls_idx} {x_c} {y_c} {w} {h}")

    label_out.write_text("\n".join(lines))


def main():
    # Build sorted class list (index = class_id - 1)
    num_classes = max(CLASS_NAMES.keys())
    class_list  = [CLASS_NAMES[i] for i in range(1, num_classes + 1)]

    converted = 0
    for src_split, (src_name, dst_name) in SPLITS.items():
        img_dir  = BASE / "images" / src_name
        lbl_dir  = BASE / "labels" / src_name

        out_img  = OUT_DIR / "images" / dst_name
        out_lbl  = OUT_DIR / "labels" / dst_name
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for xml_path in sorted(lbl_dir.glob("*.xml")):
            stem     = xml_path.stem
            img_path = img_dir / f"{stem}.jpg"

            if not img_path.exists():
                print(f"  [WARN] missing image for {xml_path.name}")
                continue

            # Copy image
            shutil.copy2(img_path, out_img / img_path.name)

            # Convert label
            convert_xml_to_yolo(
                xml_path,
                img_path,
                out_lbl / f"{stem}.txt",
                CLASS_NAMES,
            )
            converted += 1

    print(f"Converted {converted} samples → {OUT_DIR}")

    # Write dataset.yaml
    yaml_content = f"""\
path: {OUT_DIR.resolve()}
train: images/train
val:   images/val

nc: {num_classes}
names:
"""
    for name in class_list:
        yaml_content += f"  - '{name}'\n"

    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
