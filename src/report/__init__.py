"""
RoadVision AI - Civil Engineering & Maintenance Reporting Module
================================================================
Integrates Google Gemini LLM copilot for generating municipal engineering
maintenance work orders, root-cause structural analysis, and ASTM D6433 briefs.
"""

from .gemini_assistant import (
    get_gemini_api_key,
    verify_gemini_key,
    generate_civil_engineering_report
)

__all__ = [
    "get_gemini_api_key",
    "verify_gemini_key",
    "generate_civil_engineering_report"
]
