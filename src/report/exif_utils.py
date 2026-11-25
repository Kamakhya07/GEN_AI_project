"""
RoadVision AI: Authentic Image EXIF / Geotag Metadata Extractor
=============================================================
Extracts genuine GPS coordinates and capture timestamps directly from image EXIF.
Returns None if coordinates are missing. Never fabricates or estimates coordinates.
"""

from typing import Optional, Union, BinaryIO
import os
from PIL import Image, ExifTags


def extract_gps_coordinates(image_source: Union[str, Image.Image, BinaryIO]) -> Optional[str]:
    """
    Extracts authentic GPS latitude and longitude from an image's EXIF metadata.
    Returns formatted string like '18.52040° N, 73.85670° E', or None if unavailable.
    Never returns hardcoded, estimated, or fabricated GPS coordinates.
    """
    try:
        pil_img = None
        if isinstance(image_source, Image.Image):
            pil_img = image_source
        elif isinstance(image_source, str) and os.path.exists(image_source):
            pil_img = Image.open(image_source)
        elif hasattr(image_source, "read") and hasattr(image_source, "seek"):
            image_source.seek(0)
            pil_img = Image.open(image_source)

        if pil_img is None:
            return None

        exif = pil_img.getexif()
        if not exif:
            return None

        # Try standard IFD GPSInfo first (PIL >= 9.2)
        gps_ifd = None
        try:
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except Exception:
            pass

        if not gps_ifd:
            # Fallback for older dictionary mapping
            gps_ifd = exif.get(34853)

        if not gps_ifd or not isinstance(gps_ifd, dict):
            return None

        def _to_float(v):
            if isinstance(v, (tuple, list)) and len(v) == 3:
                return float(v[0]) + float(v[1]) / 60.0 + float(v[2]) / 3600.0
            elif isinstance(v, (int, float)):
                return float(v)
            return None

        lat_val = _to_float(gps_ifd.get(2))
        lat_ref = gps_ifd.get(1)
        lon_val = _to_float(gps_ifd.get(4))
        lon_ref = gps_ifd.get(3)

        if lat_val is not None and lon_val is not None and lat_ref and lon_ref:
            lat_ref_str = str(lat_ref).strip().upper()
            lon_ref_str = str(lon_ref).strip().upper()
            return f"{lat_val:.5f}° {lat_ref_str}, {lon_val:.5f}° {lon_ref_str}"
    except Exception:
        pass
    return None
