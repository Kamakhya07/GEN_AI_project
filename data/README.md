# Dataset Instructions: Road Damage Dataset (RDD2022)

## 1. Overview
The **Road Damage Dataset 2022 (RDD2022)** contains road-surface images collected from smartphone cameras mounted on car dashboards across multiple countries (Japan, India, Czech Republic, Norway, United States, China).

- **Official Repository:** [https://github.com/sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
- **Annotations:** Pascal VOC XML format (`<annotation><object><name>Dxx</name><bndbox>...`)
- **License:** Open for academic and research evaluation (CC BY-SA 4.0). Please consult the official repository for citation guidelines.

## 2. Damage Class Taxonomy
We map RDD damage codes to our five canonical classification categories:
- `D00`: **Longitudinal Crack** (Wheel mark part longitudinal crack)
- `D10`: **Transverse Crack** (Equal interval transverse crack)
- `D20`: **Alligator Crack** (Polygonal interconnected fatigue cracks)
- `D40`: **Pothole** (Cavity / depression in road surface)
- Non-annotated road surface: **Normal** (Clean background asphalt patches sampled with safety margin)

## 3. Directory Layout
Store raw downloads under:
```
data/
├── raw/
│   └── RDD2022/
│       ├── Japan/
│       │   ├── images/
│       │   └── annotations/xmls/
│       └── India/
│           ├── images/
│           └── annotations/xmls/
└── processed/
    ├── patches/
    ├── splits_manifest.json
    └── class_counts.json
```

## 4. Processing Pipeline
To parse VOC XML annotations, crop padded patches, sample normal patches, and generate zero-leakage source-level splits:
```bash
python -m src.data.process_rdd --raw_dir data/raw/RDD2022/Japan --output_dir data/processed
```
For rapid local testing and smoke tests without downloading full RDD2022:
```bash
python -m src.data.synthetic_samples
```
