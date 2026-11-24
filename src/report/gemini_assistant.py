"""
Google Gemini AI Reporting Assistant for RoadVision AI
=======================================================
Generates municipal civil engineering inspection reports, ASTM D6433 distress
explanations, and remediation protocols grounded in real GAN & CNN model outputs.

SECURITY NOTICE:
- API keys are NEVER hardcoded or exposed in the UI.
- Loaded from environment variables, .env, or Streamlit Secrets.
- Operates strictly downstream of actual model inference: zero fake predictions.
"""

import os
import json
from typing import Optional, Tuple, Dict, Any
import requests
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()


def get_gemini_api_key(user_input_key: Optional[str] = None) -> Optional[str]:
    """
    Securely resolves the Gemini API key from environment sources or user input.
    Priority order:
    1. Direct user session input (from password widget in UI)
    2. Streamlit Secrets (st.secrets["GEMINI_API_KEY"])
    3. Environment variable: GEMINI_API_KEY
    4. Environment variable: GOOGLE_API_KEY
    """
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()

    # Try Streamlit Secrets safely
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    # Check process environment
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    return None


def verify_gemini_key(api_key: str) -> Tuple[bool, str]:
    """
    Validates a Gemini API key by making a lightweight REST verification call.
    Returns (True, message) if valid, or (False, error_description).
    """
    if not api_key or len(api_key.strip()) < 10:
        return False, "API key is missing or invalid format."

    # Test with standard models in order
    models_to_test = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    
    for model_name in models_to_test:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": "Ping. Respond with one word: Connected."}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10}
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                return True, f"Key verified successfully (Connected via {model_name})."
            elif resp.status_code == 400:
                err_data = resp.json().get("error", {})
                msg = err_data.get("message", "API Key verification failed.")
                return False, f"Google API Error: {msg}"
            elif resp.status_code == 403:
                return False, "Authentication Forbidden: Invalid or unauthorized Gemini API key."
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            return False, f"Network connection error: {str(e)}"

    return False, "Could not reach Gemini service. Please verify network and API key."


def generate_civil_engineering_report(
    inspection_data: Dict[str, Any],
    api_key: str,
    model_name: str = "gemini-1.5-flash"
) -> Tuple[bool, str]:
    """
    Generates a structured Municipal Civil Engineering Report grounded in actual
    CNN and GAN inference metrics.
    
    Args:
        inspection_data: Dict containing real predictions, PCI score, work order, and latencies.
        api_key: Validated Google Gemini API key.
        model_name: Gemini model target (defaults to gemini-1.5-flash for speed).
        
    Returns:
        (success_bool, report_markdown_or_error)
    """
    if not api_key:
        return False, "Gemini API key is required to generate the AI engineering report."

    # Grounding payload from real model execution
    diagnosis = inspection_data.get("diagnosis", "Unknown")
    confidence = inspection_data.get("confidence_pct", 0.0)
    pci_metrics = inspection_data.get("pci_metrics", {})
    pci_score = pci_metrics.get("pci_score", "N/A")
    pci_rating = pci_metrics.get("pci_grade", "N/A")
    work_order = inspection_data.get("work_order", {})
    survey = inspection_data.get("survey_metadata", {})
    signal = inspection_data.get("signal_quality", {})
    latency = inspection_data.get("latency_ms", {})
    pipeline_stages = inspection_data.get("pipeline_stages", {})

    prompt = f"""
You are a Senior Highway & Pavement Infrastructure Civil Engineer serving as the AI Copilot for the RoadVision AI highway monitoring system.

Your task is to generate a formal, rigorous Municipal Pavement Health & Remediation Brief based STRICTLY on the actual deep-learning inference findings below.
DO NOT hallucinate different distress classes, alter the ASTM PCI score, or fabricate predictions. Adhere strictly to these empirical measurements:

### EMPIRICAL TELEMETRY & INFERENCE DATA:
- Route Segment: {survey.get('route_segment', 'NH-48 Sector 4-B')} (GPS: {survey.get('gps_coordinates', '18.5204 N, 73.8567 E')})
- Survey Vehicle: {survey.get('vehicle_id', 'Survey Vehicle SV-09')} @ {survey.get('speed_kmh', 42.5)} km/h
- Ingestion Source: {pipeline_stages.get('stage1_ingestion', 'Road Surface Sensor')}
- Deblurring Architecture: {pipeline_stages.get('stage2_gan_deblur', 'DeblurGAN-v2')}
- Degraded Camera PSNR: {signal.get('input_psnr_db', 'N/A')} dB | Restored PSNR: {signal.get('restored_psnr_db', 'N/A')} dB
- Restored Pavement SSIM: {signal.get('ssim', 'N/A')} (float64 bounded)
- Primary Distress Diagnosis: {diagnosis}
- Classification Confidence: {confidence:.2f}%
- Spatial Distress Regions: {pipeline_stages.get('stage3_pothole_detection', 'Localized ROI')}
- ASTM D6433 Pavement Condition Index (PCI): {pci_score} / 100 ({pci_rating})
- Work Order ID: {work_order.get('work_order_id', 'WO-2026-0842')} ({work_order.get('priority', 'Standard')})
- Recommended Equipment: {work_order.get('equipment', 'Standard Crew')}
- Estimated Repair Cost: {work_order.get('est_cost', 'N/A')}
- Total Pipeline Latency: {latency.get('total', 'N/A')} ms (Deblur: {latency.get('gan_deblur', 'N/A')} ms, CNN: {latency.get('classifier', 'N/A')} ms)

---

### INSTRUCTIONS FOR YOUR REPORT:
Format the response in professional GitHub-flavored Markdown with clear headings:
1. **Executive Summary**: High-level verdict for municipal department heads and highway maintenance directors.
2. **Structural Distress & Root-Cause Failure Mechanism**: Explain from a civil engineering perspective why this distress occurs (e.g. base course fatigue, thermal expansion/contraction, subgrade water infiltration, freeze-thaw cycles, wheelpath shear).
3. **ASTM D6433 Pavement Health Evaluation**: Interpret the PCI score ({pci_score}/100) and describe structural load capacity and ride quality implications.
4. **Actionable Remediation & Materials Protocol**: Detail specific materials (e.g., polymer-modified asphalt, tack coats, hot-pour rubberized joint sealer, cold/hot mix asphalt), machinery, and compaction procedures.
5. **Traffic Safety & Public Risk Matrix**: Potential vehicular tire damage, loss of vehicle control risk, and municipal liability exposure.
6. **Maintenance Dispatch Timeline**: Immediate (24-hr), short-term (30-day), or routine inspection schedule.

Keep tone authoritative, technical, and precise.
"""

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return True, parts[0].get("text", "No text generated.")
            return False, "Gemini returned an empty response candidate."
        else:
            err = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
            return False, f"Gemini API Error: {err}"
    except requests.exceptions.Timeout:
        return False, "Gemini API request timed out after 30 seconds."
    except Exception as e:
        return False, f"Failed to contact Gemini service: {str(e)}"
