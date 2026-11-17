import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    class_name: str
    class_id: int

@dataclass
class RDDAnnotation:
    image_id: str
    image_path: str
    width: int
    height: int
    boxes: List[BoundingBox]

# Standard RDD2022 code mappings to project damage categories
DEFAULT_RDD_MAP = {
    "D00": (1, "Longitudinal Crack"),
    "D10": (2, "Transverse Crack"),
    "D20": (3, "Alligator Crack"),
    "D40": (4, "Pothole"),
}

def parse_rdd_annotation(xml_path: str, image_dir: str, code_map: Optional[Dict[str, Tuple[int, str]]] = None) -> Optional[RDDAnnotation]:
    """
    Parses a Pascal VOC format XML annotation file from the Road Damage Dataset (RDD2022).
    """
    if code_map is None:
        code_map = DEFAULT_RDD_MAP

    if not os.path.exists(xml_path):
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_elem = root.find("filename")
        if filename_elem is not None and filename_elem.text:
            filename = filename_elem.text
        else:
            base = os.path.splitext(os.path.basename(xml_path))[0]
            filename = f"{base}.jpg"

        image_id = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, filename)

        # Image size
        size_elem = root.find("size")
        if size_elem is not None:
            width = int(size_elem.find("width").text or 0)
            height = int(size_elem.find("height").text or 0)
        else:
            width, height = 0, 0

        boxes = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                continue
            name = name_elem.text.strip()
            
            if name not in code_map:
                continue

            class_id, class_name = code_map[name]

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = int(float(bndbox.find("xmin").text or 0))
            ymin = int(float(bndbox.find("ymin").text or 0))
            xmax = int(float(bndbox.find("xmax").text or 0))
            ymax = int(float(bndbox.find("ymax").text or 0))

            if xmax > xmin and ymax > ymin:
                boxes.append(BoundingBox(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    class_name=class_name,
                    class_id=class_id
                ))

        return RDDAnnotation(
            image_id=image_id,
            image_path=image_path,
            width=width,
            height=height,
            boxes=boxes
        )
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return None
