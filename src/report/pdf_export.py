"""
RoadVision AI: Professional Municipal Civil Engineering PDF Export Engine
========================================================================
Generates formal, publication-grade municipal highway inspection PDFs adhering to
civil engineering reporting standards. Uses ONLY authentic, available data from
the actual inspection. Never invents, assumes, or fabricates values.
Unavailable fields are strictly displayed as "—".
"""

import io
import os
import time
from typing import Dict, Any, Optional, List
import cv2
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)


def _safe_val(val: Any, default: str = "—") -> str:
    """Returns the string value if non-empty, otherwise '—'."""
    if val is None:
        return default
    s = str(val).strip()
    if not s or s.lower() in ["none", "null", "n/a", "unknown", ""]:
        return default
    return s


def generate_inspection_pdf(
    report_data: Dict[str, Any],
    raw_img: Optional[np.ndarray] = None,
    processed_img: Optional[np.ndarray] = None,
    annotated_img: Optional[np.ndarray] = None,
    gemini_summary: Optional[str] = None,
    user_inputs: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Compiles a comprehensive PDF municipal pavement assessment report using strictly
    authentic, verified inspection data. All empty or unavailable fields display '—'.
    Returns bytes of the generated PDF document.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#0f172a")
    )
    sub_header_style = ParagraphStyle(
        "HeaderSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0284c7")
    )
    section_title = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )
    badge_approved = ParagraphStyle(
        "BadgeApproved",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#16a34a")
    )
    badge_pending = ParagraphStyle(
        "BadgePending",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#d97706")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#64748b")
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("ROADVISION AI &bull; MUNICIPAL PAVEMENT INSPECTION REPORT", header_style))
    elements.append(Paragraph("Civil Infrastructure Distress Localization & Structural Condition Assessment", sub_header_style))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=8))

    # Parse Authentic Metadata
    survey = report_data.get("survey_metadata", {})
    pipeline = report_data.get("pipeline_stages", {})
    pci = report_data.get("pci_metrics", {})
    wo = report_data.get("work_order", {})
    latency = report_data.get("latency_ms", {})
    stage4 = pipeline.get("stage4_detection", {}) if isinstance(pipeline.get("stage4_detection"), dict) else {}
    stage5 = pipeline.get("stage5_severity", {}) if isinstance(pipeline.get("stage5_severity"), dict) else {}
    stage6 = pipeline.get("stage6_assessment", {}) if isinstance(pipeline.get("stage6_assessment"), dict) else {}

    u_inp = user_inputs or {}

    # Route, GPS, Vehicle, Inspector
    route_val = _safe_val(u_inp.get("route_segment") or survey.get("route_segment"))
    gps_val = _safe_val(u_inp.get("gps_coordinates") or survey.get("gps_coordinates"))
    vehicle_val = _safe_val(u_inp.get("vehicle_id") or survey.get("vehicle_id"))
    inspector_val = _safe_val(u_inp.get("inspector_name") or survey.get("inspector_name"))
    timestamp_val = _safe_val(u_inp.get("timestamp") or report_data.get("timestamp"))
    ingestion_source = _safe_val(pipeline.get("stage1_ingestion"))
    processing_mode = _safe_val(pipeline.get("stage2_processing_method"))
    remarks_val = _safe_val(u_inp.get("inspection_remarks"))

    # 2. Administrative & Survey Metadata Table
    meta_table_data = [
        [
            Paragraph(f"<b>Route / Road Segment:</b> {route_val}", body_style),
            Paragraph(f"<b>GPS Location:</b> {gps_val}", body_style),
            Paragraph(f"<b>Survey Vehicle:</b> {vehicle_val}", body_style)
        ],
        [
            Paragraph(f"<b>Survey Timestamp:</b> {timestamp_val}", body_style),
            Paragraph(f"<b>Authorized Inspector:</b> {inspector_val}", body_style),
            Paragraph(f"<b>Processing Mode:</b> {processing_mode}", body_style)
        ]
    ]

    t_meta = Table(meta_table_data, colWidths=[180, 180, 180])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 8))

    # 3. Detection Results & Localization Details
    elements.append(Paragraph("1. EMPIRICAL DEFECT DETECTION & LOCALIZATION RESULTS", section_title))

    diag = str(report_data.get("diagnosis") or stage4.get("defect_class") or "—")
    conf_raw = report_data.get("confidence_pct") if report_data.get("confidence_pct") is not None else stage4.get("confidence_score")
    conf_str = f"{float(conf_raw):.1f}%" if (conf_raw is not None and float(conf_raw) > 0) else "—"

    boxes = stage4.get("bounding_boxes", [])
    box_count = len(boxes) if isinstance(boxes, list) else 0
    box_count_str = f"{box_count} Localized Defect(s)" if box_count > 0 else ("0 (Surface Clear)" if diag == "Normal" else "—")

    if boxes and isinstance(boxes, list):
        box_coords_str = ", ".join([str(b) for b in boxes[:4]])
        if len(boxes) > 4:
            box_coords_str += f" (+{len(boxes)-4} more)"
    else:
        box_coords_str = "—"

    # Latencies
    t_proc = latency.get("image_processing")
    t_cnn = latency.get("cnn_detection")
    t_tot = latency.get("total")

    if t_proc is not None and float(t_proc) > 0:
        proc_lat_str = f"{float(t_proc):.1f} ms"
    elif processing_mode in ["Direct Raw Image (GAN Bypassed)", "Original Raw Image"]:
        proc_lat_str = "Bypassed (0 ms)"
    else:
        proc_lat_str = "—"

    cnn_lat_str = f"{float(t_cnn):.1f} ms" if (t_cnn is not None and float(t_cnn) > 0) else "Not Executed"
    tot_lat_str = f"{float(t_tot):.1f} ms" if (t_tot is not None and float(t_tot) > 0) else "—"

    det_table_data = [
        [
            Paragraph("<b>Primary Detected Distress:</b>", body_style),
            Paragraph(f"<b>{diag.upper()}</b> ({conf_str} Confidence)" if diag != "—" else "—", body_style),
            Paragraph("<b>Total Defect ROIs:</b>", body_style),
            Paragraph(f"<b>{box_count_str}</b>", body_style)
        ],
        [
            Paragraph("<b>Bounding Box Coordinates:</b>", body_style),
            Paragraph(f"<code>{box_coords_str}</code>", body_style),
            Paragraph("<b>Severity Assessment:</b>", body_style),
            Paragraph(f"<b>{_safe_val(stage5.get('hazard_level'))}</b>", body_style)
        ],
        [
            Paragraph("<b>Image Processing Latency:</b>", body_style),
            Paragraph(proc_lat_str, body_style),
            Paragraph("<b>CNN Latency / Total:</b>", body_style),
            Paragraph(f"{cnn_lat_str} / {tot_lat_str}", body_style)
        ]
    ]

    t_det = Table(det_table_data, colWidths=[135, 135, 135, 135])
    t_det.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    elements.append(t_det)
    elements.append(Spacer(1, 8))

    # 4. Pavement Condition Index (PCI) — Requirement 5: Valid ASTM D6433 or "Not Available"
    elements.append(Paragraph("2. PAVEMENT CONDITION INDEX (ASTM D6433 COMPATIBILITY)", section_title))
    
    # Requirement: Show PCI only when calculated using a valid ASTM D6433 method with sufficient inspection data.
    # Single-frame frame crops lack full 225 sq m sample unit deduct density calculations.
    has_valid_astm_survey = report_data.get("astm_survey_verified", False)
    if has_valid_astm_survey:
        pci_display = f"<b>{pci.get('pci_score')} / 100</b> ({pci.get('pci_grade', 'N/A')})"
    else:
        pci_display = "<b>Not Available</b>"

    pci_condition_category = _safe_val(stage5.get("pci_condition_category"))
    pci_note = "Single camera frames evaluate localized distress severity but lack the measured 225 &plusmn; 90 m&sup2; sample unit distress densities required for certified ASTM D6433 PCI ratings. RoadVision AI does not invent certified PCI scores without full sectional survey data."

    pci_table_data = [
        [
            Paragraph("<b>ASTM D6433 PCI Score:</b>", body_style),
            Paragraph(pci_display, body_style),
            Paragraph("<b>Visual Condition Category:</b>", body_style),
            Paragraph(pci_condition_category, body_style)
        ],
        [
            Paragraph("<b>Standard Compliance Note:</b>", body_style),
            Paragraph(pci_note, disclaimer_style),
            Paragraph("", body_style),
            Paragraph("", body_style)
        ]
    ]

    t_pci = Table(pci_table_data, colWidths=[135, 135, 135, 135])
    t_pci.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("SPAN", (1, 1), (3, 1)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    elements.append(t_pci)
    elements.append(Spacer(1, 8))

    # 5. Work Order & Repair Cost Estimation in INR (₹) — Requirement 4 & 6
    elements.append(Paragraph("3. MUNICIPAL WORK ORDER & REMEDIATION SPECIFICATION", section_title))

    wo_id = _safe_val(wo.get("work_order_id") or stage6.get("work_order_id"))
    wo_priority = _safe_val(wo.get("priority") or stage6.get("priority_timeline"))
    wo_action = _safe_val(wo.get("action") or stage6.get("remediation_action"))
    wo_equipment = _safe_val(wo.get("equipment") or stage6.get("required_equipment"))

    # Work Order Approval Status (Requirement 6: Defaults to Pending Approval)
    is_approved = report_data.get("is_approved", False) or u_inp.get("approval_status") == "APPROVED & DISPATCHED"
    if is_approved:
        approved_by_str = _safe_val(report_data.get("approved_by") or u_inp.get("approved_by") or inspector_val)
        status_para = Paragraph(f"<font color='#16a34a'><b>APPROVED &amp; DISPATCHED</b></font><br/><font size='6' color='#475569'>Authorized by {approved_by_str}</font>", body_style)
    else:
        status_para = Paragraph("<font color='#d97706'><b>Pending Approval</b></font><br/><font size='6' color='#475569'>Awaiting physical engineering sign-off</font>", body_style)

    # Cost Estimation in INR (₹) (Requirement 4: INR only, clearly labeled as estimate, '—' if unavailable)
    cost_inr_str = u_inp.get("estimated_repair_cost_inr") or wo.get("est_cost_inr") or stage6.get("estimated_repair_cost_inr")
    if not cost_inr_str:
        # Fallback check if cost is configured in INR
        raw_cost = report_data.get("cost_inr") or u_inp.get("cost_inr")
        if raw_cost:
            cost_inr_str = f"₹ {raw_cost:,} (Estimated)"
        else:
            cost_inr_str = "—"

    wo_table_data = [
        [
            Paragraph("<b>Work Order ID:</b>", body_style),
            Paragraph(f"<b>{wo_id}</b>", body_style),
            Paragraph("<b>Remediation Priority:</b>", body_style),
            Paragraph(f"<b>{wo_priority}</b>", body_style)
        ],
        [
            Paragraph("<b>Approval Status:</b>", body_style),
            status_para,
            Paragraph("<b>Estimated Cost (INR):</b>", body_style),
            Paragraph(f"<b>{cost_inr_str}</b>", body_style)
        ],
        [
            Paragraph("<b>Recommended Equipment:</b>", body_style),
            Paragraph(wo_equipment, body_style),
            Paragraph("<b>Field Remarks:</b>", body_style),
            Paragraph(remarks_val, body_style)
        ],
        [
            Paragraph("<b>Remediation Protocol:</b>", body_style),
            Paragraph(wo_action, body_style),
            Paragraph("", body_style),
            Paragraph("", body_style)
        ]
    ]

    t_wo = Table(wo_table_data, colWidths=[135, 135, 135, 135])
    t_wo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("SPAN", (1, 3), (3, 3)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    elements.append(t_wo)
    elements.append(Spacer(1, 8))

    # 6. Authentic Visual Inspection Evidence (Requirement 7)
    elements.append(Paragraph("4. AUTHENTIC VISUAL INSPECTION EVIDENCE", section_title))

    def _convert_np_to_rl(img_np: np.ndarray, w: int = 170, h: int = 110) -> Optional[RLImage]:
        try:
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.ndim == 3 else img_np
            ok, buf = cv2.imencode(".png", bgr)
            if ok:
                bio = io.BytesIO(buf.tobytes())
                return RLImage(bio, width=w, height=h)
        except Exception:
            pass
        return None

    # Collect actual image figures
    figures = []
    if raw_img is not None:
        rl_raw = _convert_np_to_rl(raw_img)
        if rl_raw:
            figures.append((rl_raw, "Figure 1: Ingested Road Surface (Original)"))

    if processed_img is not None and processing_mode not in ["Direct Raw Image (GAN Bypassed)", "Original Raw Image"]:
        rl_proc = _convert_np_to_rl(processed_img)
        if rl_proc:
            figures.append((rl_proc, f"Figure 2: Restored Surface ({processing_mode})"))

    if annotated_img is not None and box_count > 0:
        rl_ann = _convert_np_to_rl(annotated_img)
        if rl_ann:
            fig_num = len(figures) + 1
            figures.append((rl_ann, f"Figure {fig_num}: YOLOv8 Spatial Localization Overlay"))

    if figures:
        if len(figures) == 3:
            t_img_data = [
                [figures[0][0], figures[1][0], figures[2][0]],
                [Paragraph(figures[0][1], disclaimer_style), Paragraph(figures[1][1], disclaimer_style), Paragraph(figures[2][1], disclaimer_style)]
            ]
            t_imgs = Table(t_img_data, colWidths=[180, 180, 180])
        elif len(figures) == 2:
            t_img_data = [
                [figures[0][0], figures[1][0]],
                [Paragraph(figures[0][1], disclaimer_style), Paragraph(figures[1][1], disclaimer_style)]
            ]
            t_imgs = Table(t_img_data, colWidths=[270, 270])
        else:
            t_img_data = [
                [figures[0][0]],
                [Paragraph(figures[0][1], disclaimer_style)]
            ]
            t_imgs = Table(t_img_data, colWidths=[540])

        t_imgs.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        elements.append(t_imgs)
        elements.append(Spacer(1, 8))
    else:
        elements.append(Paragraph("<i>Visual evidence was not recorded for this inspection session.</i>", disclaimer_style))
        elements.append(Spacer(1, 6))

    # 7. AI-Generated Engineering Recommendations (Requirement 10)
    elements.append(Paragraph("5. AI-GENERATED ADVISORY RECOMMENDATIONS", section_title))
    disclaimer_callout = Paragraph(
        "<b>ADVISORY NOTICE:</b> The following assessment is generated algorithmically from authentic CNN detection data. It serves as technical decision support and requires physical verification and sign-off by a licensed civil engineer prior to repair dispatch.",
        disclaimer_style
    )
    elements.append(disclaimer_callout)
    elements.append(Spacer(1, 4))

    if gemini_summary and len(gemini_summary.strip()) > 20:
        import re
        clean_text = gemini_summary.replace("#", "")
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_text)
        paragraphs = clean_text.strip().split("\n\n")
        for p in paragraphs[:2]:
            p_clean = p.strip().replace("\n", " ")
            if p_clean:
                elements.append(Paragraph(f"<font color='#334155'>{p_clean}</font>", body_style))
                elements.append(Spacer(1, 3))
    else:
        # Grounded procedural recommendation based strictly on actual detection
        if diag == "Pothole":
            rec_text = "Surface distress manifests localized cavity formation. Immediate cold-pour or hot-mix bituminous patching recommended to arrest moisture infiltration into the granular subbase course."
        elif diag == "Alligator Crack":
            rec_text = "Structural fatigue distress identified. Suggest asphalt mill-and-overlay or geotextile interlayer reinforcement to mitigate reflection cracking."
        elif diag == "Normal":
            rec_text = "Surface appears intact with no immediate localized cavitation. Continue routine scheduled maintenance sweeps."
        else:
            rec_text = "Surface distress detected. Standard visual pavement condition monitoring recommended."
        elements.append(Paragraph(f"<font color='#334155'>{rec_text}</font>", body_style))
        elements.append(Spacer(1, 4))

    # 8. Report Verification Footer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=5))
    elements.append(Paragraph(
        "<b>DATA INTEGRITY NOTICE:</b> Generated by RoadVision AI using authentic inspection telemetry. Only verified sensor inputs, measured latencies, and empirical detection coordinates are presented. Unverified fields are recorded as '—'.",
        disclaimer_style
    ))

    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
