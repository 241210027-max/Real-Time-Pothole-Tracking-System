from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape
import io
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# PDF reports need ReportLab:  pip install reportlab
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="RoadVision AI | Pothole Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "Models" / "best_fixed.pt"

CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Other Corruption",
    "Pothole"
]

# ----- Edit these to personalise the About page and footer -----
PROJECT_INFO = {
    "author": "Your Name",
    "institute": "Your College / University",
    "course": "B.Tech Final Year Project",
    "github": "",   # e.g. "https://github.com/your-username/roadvision-ai"
}

# UI-only data (icons, labels, links)
CLASS_ICONS = ["〰️", "✂️", "🕸️", "⚠️", "🕳️"]

PAGES = [
    "Dashboard",
    "Image Inspection",
    "Video Inspection",
    "Reports",
    "Road Guide",
    "Safety",
    "About"
]

NAV_LABELS = {
    "Dashboard": "📊 Dashboard",
    "Image Inspection": "🖼️ Image",
    "Video Inspection": "🎬 Video",
    "Reports": "📄 Reports",
    "Road Guide": "🧭 Road Guide",
    "Safety": "🛡️ Safety",
    "About": "ℹ️ About"
}

OFFICIAL_LINKS = [
    ("🏛️", "Ministry of Road Transport & Highways",
     "Policy, standards and national road programmes (India).",
     "https://morth.nic.in"),
    ("🛣️", "National Highways Authority of India",
     "Builds and maintains national highways.",
     "https://nhai.gov.in"),
    ("🏙️", "Delhi Public Works Department",
     "Roads and infrastructure maintained by Delhi PWD.",
     "https://pwd.delhi.gov.in"),
    ("🌾", "PMGSY Rural Roads",
     "Pradhan Mantri Gram Sadak Yojana rural road programme.",
     "https://pmgsy.nic.in"),
    ("🚗", "Parivahan Sewa",
     "Transport services, licences and vehicle information.",
     "https://parivahan.gov.in"),
    ("📣", "CPGRAMS Public Grievance Portal",
     "File a complaint about a damaged road with the government.",
     "https://pgportal.gov.in"),
    ("🌍", "WHO: Road traffic injuries",
     "Global facts and guidance on road safety.",
     "https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries"),
    ("🤖", "Ultralytics YOLO documentation",
     "The detection framework behind this project.",
     "https://docs.ultralytics.com"),
]

HEADER_LINKS = [
    ("MoRTH", "https://morth.nic.in"),
    ("NHAI", "https://nhai.gov.in"),
    ("Report a road issue", "https://pgportal.gov.in"),
    ("WHO Road Safety",
     "https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries"),
]

DAMAGE_GUIDE = [
    {
        "name": "Longitudinal Crack", "kind": "longitudinal",
        "what": "A crack that runs roughly parallel to the direction of traffic.",
        "cause": "Poor construction joints, fatigue under the wheel paths, or settlement at the road edge.",
        "response": "Seal early with crack sealant so water cannot get in.",
        "note": "Runs along the lane"
    },
    {
        "name": "Transverse Crack", "kind": "transverse",
        "what": "A crack that cuts across the road at roughly right angles to traffic.",
        "cause": "Temperature shrinkage of the asphalt, or a crack in a lower layer showing through.",
        "response": "Routine crack sealing, then monitor for widening.",
        "note": "Cuts across the lane"
    },
    {
        "name": "Alligator Crack", "kind": "alligator",
        "what": "A web of linked cracks that looks like reptile skin.",
        "cause": "Fatigue failure from repeated loading, often with a weak base or poor drainage.",
        "response": "Usually needs patching or replacing the damaged layers, not just sealing.",
        "note": "Interconnected mesh"
    },
    {
        "name": "Other Corruption", "kind": "other",
        "what": "Surface damage that fits no other class, such as ravelling, worn areas or failed patches.",
        "cause": "Ageing binder, water damage, utility cuts or earlier repairs breaking down.",
        "response": "Needs an on-site check to choose between a surface treatment and a repair.",
        "note": "Irregular surface wear"
    },
    {
        "name": "Pothole", "kind": "pothole",
        "what": "A bowl-shaped hole where the surface layer has broken away.",
        "cause": "Water enters cracks, weakens the layers below, and traffic breaks the loose material out.",
        "response": "Patch promptly. Deep or repeated potholes may need a larger repair.",
        "note": "Bowl-shaped hole"
    },
]


# -------------------------------------------------
# HTML HELPERS
# -------------------------------------------------

def compact(markup):
    """Strip indentation and blank lines so Streamlit's markdown
    parser never turns HTML into a code block."""
    return "\n".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )


def render(markup):
    st.markdown(compact(markup), unsafe_allow_html=True)


LOGO_SVG = """
<svg viewBox="0 0 48 48" width="42" height="42" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="rvLogo" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#14b8a6"/><stop offset="1" stop-color="#3b82f6"/>
</linearGradient></defs>
<rect width="48" height="48" rx="13" fill="url(#rvLogo)"/>
<path d="M17 40 L21.5 9 H26.5 L31 40 Z" fill="#0a1020" opacity="0.92"/>
<path d="M24 12 V17 M24 21 V27 M24 31 V37" stroke="#e6edf7" stroke-width="1.8" stroke-linecap="round"/>
<path d="M9 14 V9 H14 M34 9 H39 V14 M39 34 V39 H34 M14 39 H9 V34" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

HERO_SVG = """
<svg viewBox="0 0 420 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Road with a detected pothole">
<defs>
<linearGradient id="rvRoad" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#0f1a2f"/><stop offset="1" stop-color="#1a2a46"/>
</linearGradient>
<linearGradient id="rvScan" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="#14b8a6" stop-opacity="0"/>
<stop offset="0.5" stop-color="#14b8a6" stop-opacity="0.95"/>
<stop offset="1" stop-color="#14b8a6" stop-opacity="0"/>
</linearGradient>
</defs>
<polygon points="150,10 270,10 405,270 15,270" fill="url(#rvRoad)"/>
<path d="M150 10 L15 270 M270 10 L405 270" stroke="#3b4f72" stroke-width="2" fill="none"/>
<path class="lane" d="M210 10 V270" stroke="#e2e8f0" stroke-width="3" opacity="0.55" fill="none"/>
<path d="M170 70 L186 96 L178 124" stroke="#f59e0b" stroke-width="2" fill="none" stroke-linecap="round"/>
<path d="M262 190 L286 206 L280 230" stroke="#f59e0b" stroke-width="2" fill="none" stroke-linecap="round"/>
<ellipse cx="232" cy="176" rx="36" ry="14" fill="#050a14" stroke="#243756" stroke-width="2"/>
<circle class="ring" cx="232" cy="176" r="30" fill="none" stroke="#14b8a6" stroke-width="2"/>
<path d="M186 146 V138 H196 M268 138 H278 V146 M278 206 V214 H268 M196 214 H186 V206" fill="none" stroke="#14b8a6" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="186" y="118" width="64" height="17" rx="5" fill="#14b8a6"/>
<text x="195" y="130.5" font-size="11" font-weight="700" fill="#04211e" font-family="Inter, sans-serif">Pothole</text>
<rect class="scanline" x="15" y="10" width="390" height="3" fill="url(#rvScan)"/>
</svg>
"""


def damage_svg(kind, label, note):
    """Top-down road diagram of one damage type with a detection-style tag."""
    uid = f"rv{kind}"
    shapes = {
        "longitudinal": (
            '<path d="M96 10 L92 52 L100 92 L94 132 L98 172" '
            'stroke="#f59e0b" stroke-width="2.6" fill="none" stroke-linecap="round"/>',
            (76, 30, 44, 136)
        ),
        "transverse": (
            '<path d="M22 92 L70 88 L118 96 L168 90 L224 95 L296 91" '
            'stroke="#f59e0b" stroke-width="2.6" fill="none" stroke-linecap="round"/>',
            (16, 70, 288, 44)
        ),
        "alligator": (
            '<g stroke="#f59e0b" stroke-width="1.8" fill="none" stroke-linecap="round">'
            '<path d="M104 56 L216 52 M102 80 L218 84 M104 106 L216 110 M106 130 L214 128"/>'
            '<path d="M130 48 L134 134 M160 46 L158 136 M190 48 L194 134"/>'
            '<path d="M104 56 L134 80 M134 80 L158 56 M158 106 L190 84 M190 110 L216 130 M134 106 L158 130"/>'
            '</g>',
            (96, 44, 128, 96)
        ),
        "other": (
            '<path d="M110 70 q14 -14 34 -4 q18 8 8 26 q-14 14 -34 6 q-14 -10 -8 -28z" '
            'fill="#2b3b57" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 3"/>'
            '<path d="M170 96 q12 -10 30 -2 q14 10 2 24 q-16 8 -30 -2 z" '
            'fill="#2b3b57" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 3"/>'
            '<g fill="#f59e0b" opacity="0.8"><circle cx="156" cy="62" r="2"/>'
            '<circle cx="204" cy="76" r="2"/><circle cx="128" cy="116" r="2"/>'
            '<circle cx="214" cy="128" r="2"/><circle cx="184" cy="64" r="1.6"/></g>',
            (96, 50, 130, 90)
        ),
        "pothole": (
            '<g stroke="#f59e0b" stroke-width="2" fill="none" stroke-linecap="round">'
            '<path d="M126 84 L108 74 M194 84 L214 72 M150 70 L146 54 M176 112 L186 128"/></g>'
            '<ellipse cx="160" cy="92" rx="34" ry="22" fill="#070c18" stroke="#f43f5e" stroke-width="2.4"/>'
            '<ellipse cx="158" cy="94" rx="21" ry="12" fill="#02050c"/>',
            (100, 50, 120, 84)
        ),
    }
    body, (bx, by, bw, bh) = shapes[kind]
    tag_w = len(label) * 6.3 + 14
    return f"""
<svg viewBox="0 0 320 204" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label} diagram">
<rect width="320" height="204" rx="12" fill="#17233a"/>
<rect x="14" y="0" width="3" height="178" fill="#cbd5e1" opacity="0.35"/>
<rect x="303" y="0" width="3" height="178" fill="#cbd5e1" opacity="0.35"/>
<path class="lane slow" d="M160 0 V178" stroke="#e2e8f0" stroke-width="2.5" opacity="0.4" fill="none"/>
{body}
<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="6" fill="none" stroke="#14b8a6" stroke-width="2" stroke-dasharray="6 4"/>
<rect x="{bx}" y="{by - 17}" width="{tag_w}" height="17" rx="5" fill="#14b8a6"/>
<text x="{bx + 7}" y="{by - 5}" font-size="10.5" font-weight="700" fill="#04211e" font-family="Inter, sans-serif">{label}</text>
<path d="M0 178 H320 V192 a12 12 0 0 1 -12 12 H12 a12 12 0 0 1 -12 -12 Z" fill="#0f1a2e"/>
<text x="160" y="195" text-anchor="middle" font-size="11" fill="#9fb3cf" font-family="Inter, sans-serif">{note}</text>
<title id="{uid}">{label}</title>
</svg>
"""


CROSS_SECTION_SVG = """
<svg viewBox="0 0 640 262" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Cross-section of a road showing its layers">
<g font-family="Inter, sans-serif">
<rect x="40" y="52" width="360" height="22" fill="#2b3b57"/>
<rect x="40" y="74" width="360" height="34" fill="#34476a"/>
<rect x="40" y="108" width="360" height="44" fill="#3f5580"/>
<rect x="40" y="152" width="360" height="40" fill="#4a6396"/>
<rect x="40" y="192" width="360" height="52" fill="#5b4636"/>
<path d="M210 52 L214 66 L208 80 L213 96" stroke="#f59e0b" stroke-width="2.4" fill="none" stroke-linecap="round"/>
<path d="M300 52 L312 52 L318 60 L306 70 L296 62 Z" fill="#050a14" stroke="#f43f5e" stroke-width="1.6"/>
<g fill="#60a5fa"><circle cx="206" cy="22" r="3.2"/><circle cx="214" cy="34" r="3.2"/><circle cx="208" cy="44" r="3.2"/></g>
<path d="M222 30 H262" stroke="#60a5fa" stroke-width="1.4"/>
<text x="268" y="34" font-size="11" fill="#bcd6ff">Water enters the crack</text>
<ellipse cx="212" cy="124" rx="46" ry="20" fill="none" stroke="#f43f5e" stroke-width="1.8" stroke-dasharray="5 4"/>
<text x="266" y="128" font-size="11" fill="#fda4af">Wet base softens</text>
<text x="300" y="44" font-size="11" fill="#fda4af" text-anchor="middle">Pothole forms</text>
<g stroke="#8da2bf" stroke-width="1">
<path d="M400 63 H414 M400 91 H414 M400 130 H414 M400 172 H414 M400 218 H414"/>
</g>
<g fill="#e6edf7" font-size="12.5" font-weight="700">
<text x="422" y="62">Wearing course</text><text x="422" y="90">Base course</text>
<text x="422" y="129">Granular base</text><text x="422" y="171">Sub-base</text><text x="422" y="217">Subgrade</text>
</g>
<g fill="#8da2bf" font-size="10.5">
<text x="422" y="76">Surface layer that takes traffic</text>
<text x="422" y="104">Spreads load downward</text>
<text x="422" y="143">Main load-bearing layer</text>
<text x="422" y="185">Drainage and stability</text>
<text x="422" y="231">Natural soil underneath</text>
</g>
</g>
</svg>
"""


# -------------------------------------------------
# PREMIUM DASHBOARD CSS
# -------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --bg: #0a1020;
    --surface: #101b30;
    --surface-2: #14233b;
    --border: #1f3050;
    --border-soft: #182640;
    --text: #e6edf7;
    --muted: #8da2bf;
    --teal: #14b8a6;
    --teal-soft: rgba(20, 184, 166, 0.14);
    --blue: #3b82f6;
    --blue-soft: rgba(59, 130, 246, 0.14);
    --amber: #f59e0b;
    --red: #f43f5e;
    --green: #22c55e;
    --radius: 16px;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- App shell ---------- */

.stApp {
    background:
        radial-gradient(900px 420px at 85% -10%, rgba(20,184,166,0.10), transparent 60%),
        radial-gradient(700px 380px at -5% 0%, rgba(59,130,246,0.09), transparent 60%),
        var(--bg);
    color: var(--text);
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1320px; }

h1, h2, h3, h4 { font-family: 'Manrope', 'Inter', sans-serif; color: var(--text); letter-spacing: -0.01em; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: #243756; border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

a { color: #5eead4; }

/* ---------- Animations ---------- */

@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes lane { to { stroke-dashoffset: -26; } }
@keyframes scan { 0% { transform: translateY(0); opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { transform: translateY(255px); opacity: 0; } }
@keyframes ring { 0% { transform: scale(0.6); opacity: 0.9; } 100% { transform: scale(1.7); opacity: 0; } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.55); } 100% { box-shadow: 0 0 0 9px rgba(34,197,94,0); } }
@keyframes sweep { 0% { top: 0; opacity: 0; } 15% { opacity: 1; } 85% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
@keyframes drift { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }

.lane { stroke-dasharray: 14 12; animation: lane 1s linear infinite; }
.lane.slow { animation-duration: 2.4s; }
.scanline { animation: scan 3.8s ease-in-out infinite; }
.ring { transform-box: fill-box; transform-origin: center; animation: ring 2.2s ease-out infinite; }

/* ---------- Header ---------- */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    padding: 14px 20px;
    border-radius: 18px;
    border: 1px solid var(--border);
    background: linear-gradient(135deg, rgba(20,184,166,0.10), rgba(59,130,246,0.08)), var(--surface);
    margin-bottom: 12px;
    animation: rise 0.5s ease both;
}

.topbrand { display: flex; align-items: center; gap: 13px; }

.brand-name { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 20px; line-height: 1.1; color: #f8fafc; }
.brand-tag { font-size: 12px; color: var(--muted) !important; }

.toplinks { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }

.hlink {
    padding: 7px 13px;
    border-radius: 10px;
    font-size: 12.5px;
    font-weight: 600;
    color: #cfe0f5 !important;
    text-decoration: none !important;
    border: 1px solid var(--border);
    background: rgba(16,27,48,0.7);
    transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease;
}

.hlink:hover { border-color: var(--teal); background: var(--teal-soft); color: #fff !important; }

.live {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 13px; border-radius: 10px; font-size: 12.5px; font-weight: 600;
    background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.35); color: #bbf7d0;
}

.live.off { background: rgba(244,63,94,0.10); border-color: rgba(244,63,94,0.4); color: #fecdd3; }

.dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }
.dot.ok { background: var(--green); animation: pulse 1.8s ease-out infinite; }
.dot.err { background: var(--red); }

/* ---------- Top navigation (radio styled as pills) ---------- */

.st-key-page [role="radiogroup"] {
    flex-direction: row; flex-wrap: wrap; gap: 6px;
    padding: 6px; border-radius: 16px;
    background: var(--surface); border: 1px solid var(--border-soft);
}

.st-key-page [role="radiogroup"] label {
    border-radius: 11px; padding: 8px 16px; cursor: pointer;
    border: 1px solid transparent;
    transition: background 0.18s ease, border-color 0.18s ease;
}

.st-key-page [role="radiogroup"] label:hover { background: var(--surface-2); }
.st-key-page [role="radiogroup"] label > div:first-child { display: none; }

.st-key-page [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, var(--teal-soft), var(--blue-soft));
    border-color: rgba(20,184,166,0.5);
}

.st-key-page [role="radiogroup"] label p { font-weight: 600; font-size: 14px; }

/* ---------- Sidebar ---------- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1a30 0%, #0b1424 100%);
    border-right: 1px solid var(--border-soft);
}

[data-testid="stSidebar"] * { color: var(--text); }

.side-brand { display: flex; align-items: center; gap: 12px; padding: 4px 2px; }
.side-label { font-size: 12px; font-weight: 600; color: var(--muted) !important; margin: 4px 0 6px 2px; }

.status-pill {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px;
    border-radius: 12px; background: var(--surface); border: 1px solid var(--border-soft);
    font-size: 13px; font-weight: 500;
}

.side-link {
    display: block; padding: 8px 12px; margin-bottom: 6px; border-radius: 10px;
    font-size: 13px; font-weight: 500; text-decoration: none !important;
    color: #cfe0f5 !important; background: var(--surface); border: 1px solid var(--border-soft);
    transition: border-color 0.18s ease, background 0.18s ease;
}

.side-link:hover { border-color: var(--teal); background: var(--teal-soft); }

.side-stat { display: flex; justify-content: space-between; font-size: 13px; padding: 5px 2px; color: #c4d2e6 !important; }
.side-stat b { color: #f8fafc !important; }

/* ---------- Page headings ---------- */

.page-head { display: flex; align-items: center; gap: 16px; margin: 18px 0 6px 0; }

.page-icon {
    width: 50px; height: 50px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center; font-size: 24px;
    background: linear-gradient(135deg, var(--teal-soft), var(--blue-soft));
    border: 1px solid rgba(20,184,166,0.35);
}

.main-title { font-family: 'Manrope', sans-serif; font-size: 31px; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; line-height: 1.15; }
.subtitle { color: var(--muted); font-size: 14.5px; margin-bottom: 22px; margin-top: 4px; }

.section-title {
    font-family: 'Manrope', sans-serif; color: #f8fafc; font-size: 19px; font-weight: 700;
    margin-top: 28px; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;
}

.section-title::before { content: ""; width: 4px; height: 20px; border-radius: 4px; background: linear-gradient(180deg, var(--teal), var(--blue)); }

.panel-title { font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 15px; color: #f1f5fb; margin-bottom: 10px; }

/* ---------- Hero ---------- */

.hero {
    display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap;
    padding: 34px 38px; border-radius: 22px; border: 1px solid var(--border); margin: 8px 0 20px 0;
    background:
        radial-gradient(520px 240px at 100% 0%, rgba(20,184,166,0.22), transparent 70%),
        radial-gradient(420px 220px at 0% 100%, rgba(59,130,246,0.18), transparent 70%),
        linear-gradient(145deg, #13213a, #0f1a2f);
    animation: rise 0.6s ease 0.1s both;
}

.hero-text { flex: 1 1 380px; }
.hero h2 { margin: 0 0 12px 0; font-size: 34px; font-weight: 800; line-height: 1.15; }
.hero p { margin: 0; max-width: 560px; color: #a9bbd4; font-size: 15.5px; line-height: 1.65; }
.hero-art { flex: 0 1 400px; min-width: 260px; }
.hero-art svg { width: 100%; height: auto; display: block; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
.chip { padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; color: #bfeee8; background: var(--teal-soft); border: 1px solid rgba(20,184,166,0.35); }
.chip.blue { color: #bcd6ff; background: var(--blue-soft); border-color: rgba(59,130,246,0.4); }

/* ---------- Cards ---------- */

.metric-card {
    position: relative; background: linear-gradient(145deg, #15253f, #101b30);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 20px 20px 18px 22px; min-height: 118px; overflow: hidden;
    transition: transform 0.18s ease, border-color 0.18s ease;
}

.metric-card:hover { transform: translateY(-2px); border-color: rgba(20,184,166,0.5); }
.metric-card::after { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent, var(--teal)); }
.metric-card.tone-blue { --accent: var(--blue); }
.metric-card.tone-red { --accent: var(--red); }
.metric-card.tone-amber { --accent: var(--amber); }
.metric-card.tone-teal { --accent: var(--teal); }

.metric-label { color: var(--muted); font-size: 13px; font-weight: 500; }
.metric-value { font-family: 'Manrope', sans-serif; color: #f8fafc; font-size: 31px; font-weight: 800; margin-top: 8px; line-height: 1.1; }
.metric-hint { color: #6f86a6; font-size: 12px; margin-top: 6px; }

.info-card {
    background: linear-gradient(145deg, #14233b, #101b30); border: 1px solid var(--border);
    padding: 22px; border-radius: var(--radius); margin-bottom: 12px; height: 100%;
}

.info-card h3 { margin: 0 0 8px 0; font-size: 18px; }
.info-card p, .info-card li { color: #a1b4cf; font-size: 14px; line-height: 1.65; }
.info-card p { margin: 0; }
.info-card ul { margin: 6px 0 0 0; padding-left: 18px; }

.info-card.warn { border-color: rgba(245,158,11,0.5); background: linear-gradient(145deg, rgba(245,158,11,0.10), #101b30); }
.info-card.danger { border-color: rgba(244,63,94,0.5); background: linear-gradient(145deg, rgba(244,63,94,0.10), #101b30); }

.link-card {
    display: block; text-decoration: none !important; height: 100%;
    background: linear-gradient(145deg, #14233b, #101b30); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px; margin-bottom: 12px;
    transition: transform 0.18s ease, border-color 0.18s ease;
}

.link-card:hover { transform: translateY(-2px); border-color: var(--teal); }
.link-card .lc-icon { font-size: 22px; }
.link-card .lc-title { font-weight: 700; font-size: 14.5px; color: #f1f5fb; margin-top: 8px; }
.link-card .lc-desc { font-size: 12.5px; color: var(--muted); margin-top: 4px; line-height: 1.5; }
.link-card .lc-go { font-size: 12px; color: #5eead4; margin-top: 10px; font-weight: 600; }

.class-card {
    background: linear-gradient(145deg, #15253f, #101b30); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px 16px; min-height: 128px;
    transition: transform 0.18s ease, border-color 0.18s ease;
}

.class-card:hover { transform: translateY(-2px); border-color: rgba(59,130,246,0.55); }
.class-icon { width: 40px; height: 40px; border-radius: 11px; display: flex; align-items: center; justify-content: center; font-size: 19px; background: var(--blue-soft); margin-bottom: 12px; }
.class-name { font-size: 14px; font-weight: 700; color: #f1f5fb; }
.class-tag { font-size: 12px; color: var(--muted); margin-top: 2px; }

.guide-card {
    background: linear-gradient(145deg, #14233b, #101b30); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 14px 14px 18px 14px; margin-bottom: 14px; height: 100%;
}

.guide-card svg { width: 100%; height: auto; display: block; border-radius: 12px; }
.guide-card h4 { margin: 14px 0 6px 0; font-size: 16px; }
.guide-card p { margin: 0 0 8px 0; font-size: 13.5px; line-height: 1.6; color: #a9bbd4; }
.guide-card b { color: #e6edf7; }

.svg-card { background: linear-gradient(145deg, #14233b, #101b30); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
.svg-card svg { width: 100%; height: auto; display: block; }

.step { display: flex; gap: 14px; align-items: flex-start; padding: 14px 16px; border-radius: 14px; background: var(--surface); border: 1px solid var(--border-soft); margin-bottom: 10px; }
.step-num { width: 28px; height: 28px; border-radius: 9px; flex: 0 0 auto; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; color: #04211e; background: linear-gradient(135deg, var(--teal), #5eead4); }
.step-title { font-weight: 600; font-size: 14px; }
.step-text { color: var(--muted); font-size: 13px; margin-top: 2px; }

.flow { display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; }
.flow-node { flex: 1 1 130px; padding: 14px; border-radius: 14px; background: var(--surface); border: 1px solid var(--border); text-align: center; }
.flow-node .fi { font-size: 22px; }
.flow-node .ft { font-weight: 700; font-size: 13.5px; margin-top: 6px; }
.flow-node .fd { font-size: 12px; color: var(--muted); margin-top: 3px; }

.empty-state {
    position: relative; overflow: hidden; border: 1.5px dashed var(--border); border-radius: var(--radius);
    background: rgba(16,27,48,0.55); padding: 54px 20px; text-align: center; color: var(--muted); font-size: 14px;
}

.empty-state::after {
    content: ""; position: absolute; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    animation: sweep 3.2s ease-in-out infinite;
}

.empty-state .big { font-size: 34px; margin-bottom: 8px; }

.file-chip { display: inline-flex; align-items: center; gap: 10px; padding: 8px 14px; border-radius: 12px; background: var(--surface); border: 1px solid var(--border-soft); font-size: 13px; margin: 8px 0 14px 0; }

.sev-wrap { margin-top: 16px; padding: 16px 18px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border-soft); }
.sev-bar { display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: #1a2a45; margin: 10px 0 12px 0; }
.sev-seg.low { background: var(--green); }
.sev-seg.med { background: var(--amber); }
.sev-seg.high { background: var(--red); }
.sev-legend { display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px; color: #c4d2e6; }
.sev-legend span i { display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 7px; }

.hist-card { padding: 14px 18px; border-radius: 14px; background: var(--surface); border: 1px solid var(--border-soft); }
.hist-title { font-weight: 700; font-size: 14.5px; }
.hist-meta { font-size: 12.5px; color: var(--muted); margin-top: 3px; }

.sos { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
.sos div { flex: 1 1 150px; padding: 12px 14px; border-radius: 12px; background: rgba(244,63,94,0.10); border: 1px solid rgba(244,63,94,0.35); font-size: 13px; color: #fecdd3; }
.sos b { display: block; font-family: 'Manrope', sans-serif; font-size: 22px; color: #fff; }

/* ---------- Footer ---------- */

.site-footer { margin-top: 44px; padding: 26px 28px 18px 28px; border-radius: 20px; border: 1px solid var(--border-soft); background: linear-gradient(180deg, #0f1a2f, #0b1424); }
.foot-grid { display: flex; flex-wrap: wrap; gap: 28px; }
.foot-col { flex: 1 1 200px; }
.foot-col h5 { margin: 0 0 10px 0; font-family: 'Manrope', sans-serif; font-size: 13.5px; color: #f1f5fb; }
.foot-col a { display: block; font-size: 13px; color: #9fb3cf !important; text-decoration: none !important; margin-bottom: 7px; }
.foot-col a:hover { color: #5eead4 !important; }
.foot-col p { font-size: 12.5px; color: #8da2bf; line-height: 1.6; margin: 8px 0 0 0; }
.foot-bottom { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border-soft); font-size: 12px; color: #6f86a6; }

/* ---------- Streamlit widgets ---------- */

.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 600; min-height: 46px; border: 1px solid transparent;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
}

.stButton > button { background: linear-gradient(90deg, #0d9488, #2563eb); color: white; box-shadow: 0 8px 22px rgba(13,148,136,0.28); }
.stButton > button:hover { color: white; border-color: transparent; filter: brightness(1.1); transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

.stDownloadButton > button { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); }
.stDownloadButton > button:hover { color: #fff; border-color: var(--teal); background: var(--teal-soft); }

[data-testid="stFileUploaderDropzone"] { background: rgba(16,27,48,0.7); border: 1.5px dashed #2a4268; border-radius: var(--radius); padding: 22px; transition: border-color 0.18s ease, background 0.18s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--teal); background: rgba(20,184,166,0.06); }
[data-testid="stFileUploaderDropzone"] button { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 10px; }

[data-testid="stImage"] img, [data-testid="stVideo"] video { border-radius: 14px; border: 1px solid var(--border); }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--border); }
[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, var(--teal), var(--blue)); }
[data-testid="stSlider"] [role="slider"] { background: var(--teal); box-shadow: 0 0 0 4px rgba(20,184,166,0.2); }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border-soft); }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; border-radius: 10px 10px 0 0; }
.stTabs [aria-selected="true"] { color: #5eead4; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--teal); }

[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: var(--surface); }

h3 { font-size: 19px; }
hr { border-color: var(--border-soft); }

/* ---------- Responsive ---------- */

@media (max-width: 900px) {
    .block-container { padding: 1rem 0.9rem 2rem 0.9rem; }
    .main-title { font-size: 25px; }
    .hero { padding: 24px 20px; }
    .hero h2 { font-size: 25px; }
    .metric-value { font-size: 27px; }
    .page-icon { width: 42px; height: 42px; font-size: 20px; }
    .toplinks .hlink { display: none; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

st.session_state.setdefault("page", "Dashboard")
st.session_state.setdefault("history", [])


def go_to(page_name):
    st.session_state["page"] = page_name


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# -------------------------------------------------
# DETECTION UTILITIES
# -------------------------------------------------

def get_severity(box, image_width, image_height):
    """
    Approximate severity using bounding-box area.
    This is an image-area heuristic, not a real
    measurement of pothole depth or road danger.
    """

    x1, y1, x2, y2 = box

    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    image_area = image_width * image_height

    if image_area == 0:
        return "Unknown"

    percentage = (box_area / image_area) * 100

    if percentage >= 8:
        return "High"
    elif percentage >= 2.5:
        return "Medium"
    else:
        return "Low"


def extract_detections(result, image_width, image_height):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        coordinates = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        x1, y1, x2, y2 = coordinates

        if isinstance(result.names, dict):
            class_name = result.names.get(
                class_id, f"Class {class_id}"
            )
        else:
            class_name = result.names[class_id]

        severity = get_severity(
            coordinates,
            image_width,
            image_height
        )

        detections.append({
            "Class": class_name,
            "Confidence": round(confidence * 100, 2),
            "Severity": severity,
            "X1": round(float(x1), 1),
            "Y1": round(float(y1), 1),
            "X2": round(float(x2), 1),
            "Y2": round(float(y2), 1)
        })

    return detections


def summarize(detections):
    """Counts used by reports and history (does not alter detections)."""
    classes = {}
    for item in detections:
        classes[item["Class"]] = classes.get(item["Class"], 0) + 1

    return {
        "total": len(detections),
        "potholes": sum(1 for d in detections if "pothole" in d["Class"].lower()),
        "high": sum(1 for d in detections if d["Severity"] == "High"),
        "medium": sum(1 for d in detections if d["Severity"] == "Medium"),
        "low": sum(1 for d in detections if d["Severity"] == "Low"),
        "classes": classes
    }


# -------------------------------------------------
# UI HELPERS (presentation only)
# -------------------------------------------------

def page_header(icon, title, subtitle):
    render(
        f'<div class="page-head"><div class="page-icon">{icon}</div>'
        f'<div class="main-title">{title}</div></div>'
        f'<div class="subtitle">{subtitle}</div>'
    )


def section_title(text):
    render(f'<div class="section-title">{text}</div>')


def metric_card(label, value, hint="", tone="teal"):
    return (
        f'<div class="metric-card tone-{tone}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-hint">{hint}</div>'
        f'</div>'
    )


def empty_state(icon, text):
    render(f'<div class="empty-state"><div class="big">{icon}</div>{text}</div>')


def link_cards(items, columns=4):
    for start in range(0, len(items), columns):
        row = items[start:start + columns]
        cols = st.columns(columns)
        for col, (icon, title, desc, url) in zip(cols, row):
            col.markdown(
                compact(
                    f'<a class="link-card" href="{url}" target="_blank" rel="noopener noreferrer">'
                    f'<div class="lc-icon">{icon}</div>'
                    f'<div class="lc-title">{title}</div>'
                    f'<div class="lc-desc">{desc}</div>'
                    f'<div class="lc-go">Open official site ↗</div></a>'
                ),
                unsafe_allow_html=True
            )


def detection_table_config():
    return {
        "Class": st.column_config.TextColumn(
            "Class", help="Detected road damage type"
        ),
        "Confidence": st.column_config.ProgressColumn(
            "Confidence",
            help="Model confidence for this detection",
            min_value=0,
            max_value=100,
            format="%.1f%%"
        ),
        "Severity": st.column_config.TextColumn(
            "Severity",
            help="Estimated from bounding-box area only"
        )
    }


def show_metrics(detections):
    total = len(detections)

    potholes = sum(
        1 for item in detections
        if "pothole" in item["Class"].lower()
    )

    high = sum(
        1 for item in detections
        if item["Severity"] == "High"
    )

    medium = sum(
        1 for item in detections
        if item["Severity"] == "Medium"
    )

    low = sum(
        1 for item in detections
        if item["Severity"] == "Low"
    )

    columns = st.columns(4)

    metrics = [
        ("Total Detections", total, "All damage found", "teal"),
        ("Potholes", potholes, "Pothole class only", "blue"),
        ("High Severity", high, "Large visible area", "red"),
        ("Medium Severity", medium, "Moderate visible area", "amber")
    ]

    for column, (label, value, hint, tone) in zip(columns, metrics):
        column.markdown(
            compact(metric_card(label, value, hint, tone)),
            unsafe_allow_html=True
        )

    bar_total = low + medium + high
    low_w = (low / bar_total * 100) if bar_total else 0
    med_w = (medium / bar_total * 100) if bar_total else 0
    high_w = (high / bar_total * 100) if bar_total else 0

    render(
        f'<div class="sev-wrap">'
        f'<div class="panel-title" style="margin-bottom:0;">Severity distribution</div>'
        f'<div class="sev-bar">'
        f'<div class="sev-seg low" style="width:{low_w}%"></div>'
        f'<div class="sev-seg med" style="width:{med_w}%"></div>'
        f'<div class="sev-seg high" style="width:{high_w}%"></div>'
        f'</div>'
        f'<div class="sev-legend">'
        f'<span><i style="background:#22c55e"></i>Low: {low}</span>'
        f'<span><i style="background:#f59e0b"></i>Medium: {medium}</span>'
        f'<span><i style="background:#f43f5e"></i>High: {high}</span>'
        f'</div></div>'
    )


def show_class_chart(detections):
    if not detections:
        return
    counts = pd.Series([d["Class"] for d in detections]).value_counts()
    st.bar_chart(counts, color="#14b8a6", height=240)


# -------------------------------------------------
# PDF REPORTS
# -------------------------------------------------

def _latin(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _para(text):
    return escape(_latin(text))


def array_to_jpeg(rgb_array, max_side=1400):
    """Encode an RGB array to JPEG bytes for PDF embedding."""
    img = Image.fromarray(rgb_array)
    img.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), img.width, img.height


def encode_frame(frame_bgr, max_width=800):
    height, width = frame_bgr.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame_bgr = cv2.resize(
            frame_bgr, (int(width * scale), int(height * scale))
        )
        height, width = frame_bgr.shape[:2]
    ok, buffer = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    if not ok:
        return None
    return buffer.tobytes(), width, height


def _pdf_table(rows, widths, font=8.5, header=True, zebra=True):
    safe_rows = [[_latin(cell) for cell in row] for row in rows]
    table = Table(safe_rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    if zebra:
        for i in range(1 if header else 0, len(safe_rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f1f5f9")))
    table.setStyle(TableStyle(style))
    return table


def _pdf_page_decor(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#0b1424"))
    canvas.rect(0, height - 18 * mm, width, 18 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#14b8a6"))
    canvas.rect(0, height - 18 * mm, width, 1.2 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(16 * mm, height - 11.5 * mm, "RoadVision AI")
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#9fb3cf"))
    canvas.drawRightString(width - 16 * mm, height - 11.5 * mm, "Road Damage Detection Report")
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(16 * mm, 10 * mm, "Automated AI estimate. Not a substitute for a professional road inspection.")
    canvas.drawRightString(width - 16 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_report(title, subtitle, meta_rows, summary, images, dataframe, max_rows=40):
    """Generic report builder used for both image and video results."""
    if not REPORTLAB_OK:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=26 * mm, bottomMargin=18 * mm,
        title=_latin(title), author="RoadVision AI"
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("rv_h1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                        fontSize=19, textColor=colors.HexColor("#0f172a"), spaceAfter=2)
    h2 = ParagraphStyle("rv_h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                        fontSize=12.5, textColor=colors.HexColor("#0f766e"),
                        spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("rv_body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=9.5, leading=13.5, textColor=colors.HexColor("#334155"))
    small = ParagraphStyle("rv_small", parent=body, fontSize=8.5, leading=12,
                           textColor=colors.HexColor("#64748b"))

    full_width = 178 * mm
    story = [
        Paragraph(_para(title), h1),
        Paragraph(_para(subtitle), body),
        Spacer(1, 8),
        _pdf_table([[k, v] for k, v in meta_rows], [48 * mm, 130 * mm],
                   header=False, zebra=True),
        Paragraph("Summary", h2),
        _pdf_table(
            [["Total detections", "Potholes", "High severity", "Medium severity", "Low severity"],
             [summary["total"], summary["potholes"], summary["high"],
              summary["medium"], summary["low"]]],
            [full_width / 5] * 5, font=10.5, zebra=False
        ),
        Spacer(1, 8),
    ]

    if summary["classes"]:
        rows = [["Damage class", "Count"]] + [
            [name, count] for name, count in
            sorted(summary["classes"].items(), key=lambda x: -x[1])
        ]
        story.append(_pdf_table(rows, [120 * mm, 58 * mm]))
    else:
        story.append(Paragraph(
            "No road damage was detected at the selected confidence threshold.", body))

    if images:
        story.append(Paragraph("Annotated results", h2))
        for caption, jpeg, width, height in images:
            scale = min(170 * mm / width, 105 * mm / height)
            story.append(KeepTogether([
                RLImage(io.BytesIO(jpeg), width=width * scale, height=height * scale),
                Paragraph(_para(caption), small),
                Spacer(1, 8),
            ]))

    if dataframe is not None and len(dataframe) > 0:
        story.append(Paragraph("Detection details", h2))
        shown = dataframe.head(max_rows)
        columns = list(shown.columns)
        rows = [columns] + shown.astype(str).values.tolist()
        story.append(_pdf_table(rows, [full_width / len(columns)] * len(columns), font=7.5))
        if len(dataframe) > max_rows:
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"Showing the first {max_rows} of {len(dataframe)} detections. "
                f"Download the CSV report for the full list.", small))

    story += [
        Paragraph("Notes and disclaimer", h2),
        Paragraph(
            "Severity is estimated only from the size of each bounding box relative to the "
            "image or frame (Low below 2.5%, Medium from 2.5% to 8%, High at 8% or more). "
            "It does not measure pothole depth, physical size or the actual danger to road users.",
            body),
        Spacer(1, 4),
        Paragraph(
            "This report is produced automatically by an AI model and can miss damage or flag "
            "damage that is not there. It is not a substitute for an inspection by a qualified "
            "engineer and must not be used as the only basis for safety, maintenance or legal "
            "decisions. RoadVision AI does not predict or prevent road accidents, and its authors "
            "accept no liability for accidents, injury or damage arising from its use.",
            body),
    ]

    doc.build(story, onFirstPage=_pdf_page_decor, onLaterPages=_pdf_page_decor)
    return buffer.getvalue()


def safe_pdf(*args, **kwargs):
    """Build a PDF but never let a report problem break the analysis."""
    try:
        return build_pdf_report(*args, **kwargs)
    except Exception as error:
        st.warning(f"The PDF report could not be generated: {error}")
        return None


def pdf_download_button(pdf_bytes, file_name, key):
    if pdf_bytes:
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            key=key,
            use_container_width=True
        )
    elif not REPORTLAB_OK:
        st.info("PDF reports need ReportLab. Run: pip install reportlab")


def add_history(kind, name, detections, pdf_bytes, csv_bytes):
    stats = summarize(detections)
    st.session_state["history"].insert(0, {
        "kind": kind,
        "name": name,
        "time": datetime.now().strftime("%d %b %Y, %H:%M"),
        "stats": stats,
        "pdf": pdf_bytes,
        "csv": csv_bytes,
    })
    del st.session_state["history"][20:]


# -------------------------------------------------
# HEADER, SIDEBAR, FOOTER
# -------------------------------------------------

def render_header(model_ok):
    links = "".join(
        f'<a class="hlink" href="{url}" target="_blank" rel="noopener noreferrer">{name} ↗</a>'
        for name, url in HEADER_LINKS
    )
    status = (
        '<span class="live"><span class="dot ok"></span>Model ready</span>'
        if model_ok else
        '<span class="live off"><span class="dot err"></span>Model offline</span>'
    )
    render(
        f'<div class="topbar"><div class="topbrand">{LOGO_SVG}'
        f'<div><div class="brand-name">RoadVision AI</div>'
        f'<div class="brand-tag">Intelligent road damage detection</div></div></div>'
        f'<div class="toplinks">{status}{links}</div></div>'
    )


def render_footer():
    year = datetime.now().year
    github = (
        f'<a href="{PROJECT_INFO["github"]}" target="_blank" rel="noopener noreferrer">Project source code ↗</a>'
        if PROJECT_INFO["github"] else ""
    )
    render(
        f'<div class="site-footer"><div class="foot-grid">'
        f'<div class="foot-col"><div class="topbrand">{LOGO_SVG}'
        f'<div><div class="brand-name">RoadVision AI</div></div></div>'
        f'<p>AI-assisted detection of potholes and road cracks from photos and video, '
        f'with severity estimates and downloadable reports.</p></div>'
        f'<div class="foot-col"><h5>Official resources</h5>'
        f'<a href="https://morth.nic.in" target="_blank" rel="noopener noreferrer">MoRTH, India ↗</a>'
        f'<a href="https://nhai.gov.in" target="_blank" rel="noopener noreferrer">NHAI ↗</a>'
        f'<a href="https://pgportal.gov.in" target="_blank" rel="noopener noreferrer">Report a road issue (CPGRAMS) ↗</a>'
        f'<a href="https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries" target="_blank" rel="noopener noreferrer">WHO road safety ↗</a></div>'
        f'<div class="foot-col"><h5>Project</h5>'
        f'<a href="https://docs.ultralytics.com" target="_blank" rel="noopener noreferrer">Ultralytics YOLO ↗</a>'
        f'<a href="https://docs.streamlit.io" target="_blank" rel="noopener noreferrer">Streamlit ↗</a>'
        f'{github}'
        f'<p>{PROJECT_INFO["course"]}<br>{PROJECT_INFO["institute"]}</p></div>'
        f'<div class="foot-col"><h5>Disclaimer</h5>'
        f'<p>Results are automated estimates and may be wrong. They are not a substitute '
        f'for a professional inspection and do not predict or prevent road accidents. '
        f'Never use this app while driving.</p></div></div>'
        f'<div class="foot-bottom"><span>© {year} RoadVision AI. All rights reserved.</span>'
        f'<span>Built with Streamlit, YOLO and OpenCV</span></div></div>'
    )


with st.sidebar:
    render(f'<div class="side-brand">{LOGO_SVG}'
           f'<div><div class="brand-name">RoadVision AI</div>'
           f'<div class="brand-tag">Road damage detection</div></div></div>')
    st.divider()

    st.markdown('<div class="side-label">Detection settings</div>',
                unsafe_allow_html=True)

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.01,
        max_value=1.00,
        value=0.35,
        step=0.01,
        format="%.2f",
        help="Detections below this score are ignored. "
             "Lower it to find more damage, raise it to reduce false alarms."
    )

    st.caption("Adjust the minimum model confidence.")

    st.divider()

    st.markdown('<div class="side-label">System status</div>',
                unsafe_allow_html=True)
    sidebar_status = st.empty()

    st.divider()

    history = st.session_state["history"]
    st.markdown('<div class="side-label">This session</div>',
                unsafe_allow_html=True)
    render(
        f'<div class="side-stat"><span>Analyses run</span><b>{len(history)}</b></div>'
        f'<div class="side-stat"><span>Detections found</span>'
        f'<b>{sum(h["stats"]["total"] for h in history)}</b></div>'
        f'<div class="side-stat"><span>Reports ready</span>'
        f'<b>{sum(1 for h in history if h["pdf"])}</b></div>'
    )

    st.divider()

    st.markdown('<div class="side-label">Official links</div>',
                unsafe_allow_html=True)
    render("".join(
        f'<a class="side-link" href="{url}" target="_blank" rel="noopener noreferrer">{name} ↗</a>'
        for name, url in HEADER_LINKS
    ))

    st.divider()
    st.caption("B.Tech Project | Pothole Detection")


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model_error = None

try:
    model = load_model()
    model_status = True
except Exception as error:
    model_status = False
    model_error = error

if model_status:
    sidebar_status.markdown(
        '<div class="status-pill"><span class="dot ok"></span>Model ready</div>',
        unsafe_allow_html=True
    )
else:
    sidebar_status.markdown(
        '<div class="status-pill"><span class="dot err"></span>Model unavailable</div>',
        unsafe_allow_html=True
    )


# -------------------------------------------------
# TOP HEADER AND NAVIGATION
# -------------------------------------------------

render_header(model_status)

page = st.radio(
    "NAVIGATION",
    PAGES,
    horizontal=True,
    key="page",
    format_func=lambda option: NAV_LABELS[option],
    label_visibility="collapsed"
)

if model_error is not None:
    st.error("Unable to load the YOLO model.")
    st.exception(model_error)


# -------------------------------------------------
# IMAGE WORKFLOW (shared by Dashboard and Image page)
# -------------------------------------------------

def image_workflow(prefix):
    state_key = f"{prefix}_image_result"

    uploaded_image = st.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"{prefix}_image_upload",
        help="Supported formats: JPG, JPEG, PNG, WEBP"
    )

    if uploaded_image is None:
        empty_state(
            "🖼️",
            "No image yet. Drop a road photo above to get started."
        )
        return

    if not model_status:
        st.warning("The model is not ready, so analysis is unavailable.")
        return

    file_id = (uploaded_image.name, uploaded_image.size)

    image = Image.open(uploaded_image).convert("RGB")
    image_array = np.array(image)

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        render('<div class="panel-title">Input image</div>')

        st.image(
            image,
            caption="Uploaded Road Image",
            use_container_width=True
        )

        detect_clicked = st.button(
            "🔍 Detect Road Damage",
            use_container_width=True,
            key=f"{prefix}_image_detect"
        )

    if detect_clicked:

        with right_col:
            with st.spinner("Analyzing image..."):

                results = model.predict(
                    source=image_array,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                annotated = result.plot()
                annotated_rgb = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                detections = extract_detections(
                    result,
                    image.width,
                    image.height
                )

        dataframe = pd.DataFrame(detections) if detections else None
        csv_bytes = (
            dataframe.to_csv(index=False).encode("utf-8")
            if dataframe is not None else None
        )

        jpeg, jw, jh = array_to_jpeg(annotated_rgb)
        pdf_bytes = safe_pdf(
            "Image Inspection Report",
            f"Road damage analysis of {uploaded_image.name}",
            [
                ("File", uploaded_image.name),
                ("Analysed on", datetime.now().strftime("%d %b %Y, %H:%M")),
                ("Image size", f"{image.width} x {image.height} px"),
                ("Confidence threshold", f"{confidence:.2f}"),
                ("Model", MODEL_PATH.name),
            ],
            summarize(detections),
            [("Annotated image with detected road damage", jpeg, jw, jh)],
            dataframe
        )

        st.session_state[state_key] = {
            "file_id": file_id,
            "name": uploaded_image.name,
            "annotated_rgb": annotated_rgb,
            "detections": detections,
            "csv": csv_bytes,
            "pdf": pdf_bytes,
        }
        add_history("Image", uploaded_image.name, detections, pdf_bytes, csv_bytes)
        st.toast("Image analysis completed", icon="✅")

    res = st.session_state.get(state_key)
    matched = bool(res) and res["file_id"] == file_id

    with right_col:
        render('<div class="panel-title">Detection output</div>')
        if matched:
            st.image(
                res["annotated_rgb"],
                caption="Detected Road Damage",
                use_container_width=True
            )
        else:
            empty_state(
                "🔍",
                "Results will appear here.<br>Click <b>Detect Road Damage</b> to start."
            )

    if matched:
        if detect_clicked:
            st.success("Image analysis completed.")

        section_title("Detection Results")

        show_metrics(res["detections"])

        if res["detections"]:
            chart_col, table_col = st.columns([1, 2], gap="large")

            with chart_col:
                render('<div class="panel-title">Damage by class</div>')
                show_class_chart(res["detections"])

            with table_col:
                render('<div class="panel-title">Detection details</div>')
                st.dataframe(
                    pd.DataFrame(res["detections"]),
                    use_container_width=True,
                    hide_index=True,
                    column_config=detection_table_config()
                )
        else:
            st.info("No road damage detected in this image.")

        section_title("Download Reports")

        dl1, dl2 = st.columns(2)

        with dl1:
            pdf_download_button(
                res["pdf"], "roadvision_image_report.pdf", f"{prefix}_image_pdf"
            )

        with dl2:
            if res["csv"]:
                st.download_button(
                    "⬇️ Download Detection Report (CSV)",
                    data=res["csv"],
                    file_name="road_damage_report.csv",
                    mime="text/csv",
                    key=f"{prefix}_image_csv",
                    use_container_width=True
                )


# -------------------------------------------------
# VIDEO WORKFLOW (shared by Dashboard and Video page)
# -------------------------------------------------

def process_video(uploaded_video, file_id):
    """Frame-by-frame analysis. Returns a result dict, or None on failure."""

    temp_input = None
    temp_output = None
    cap = None
    writer = None
    result_data = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_video.name).suffix
        ) as input_file:
            input_file.write(uploaded_video.getbuffer())
            temp_input = input_file.name

        cap = cv2.VideoCapture(temp_input)

        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid video dimensions.")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as output_file:
            temp_output = output_file.name

        writer = cv2.VideoWriter(
            temp_output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError("Could not create output video.")

        section_title("Live Analysis")

        progress = st.progress(0)
        status = st.empty()
        preview = st.empty()

        frame_number = 0
        total_detections = 0
        all_detections = []
        frame_counts = []
        key_frames = []   # a few frames with the most detections, for the PDF

        start_time = time.time()

        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model.predict(
                source=frame,
                conf=confidence,
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            writer.write(annotated)

            frame_detections = extract_detections(
                result,
                width,
                height
            )

            total_detections += len(frame_detections)
            frame_counts.append(len(frame_detections))

            for detection in frame_detections:
                detection["Frame"] = frame_number + 1
                all_detections.append(detection)

            if frame_detections and (
                len(key_frames) < 6 or len(frame_detections) > key_frames[-1][0]
            ):
                encoded = encode_frame(annotated)
                if encoded:
                    key_frames.append(
                        (len(frame_detections), frame_number + 1) + encoded
                    )
                    key_frames.sort(key=lambda item: -item[0])
                    del key_frames[6:]

            frame_number += 1

            if frame_number % 10 == 0 or frame_number == total_frames:
                progress_value = (
                    frame_number / total_frames
                    if total_frames > 0 else 0
                )

                progress.progress(min(progress_value, 1.0))

                status.write(
                    f"Processed {frame_number} frames | "
                    f"Detections: {total_detections}"
                )

                preview.image(
                    cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption=f"Frame {frame_number}",
                    use_container_width=True
                )

        cap.release()
        cap = None

        writer.release()
        writer = None

        elapsed = time.time() - start_time

        with open(temp_output, "rb") as video_file:
            video_bytes = video_file.read()

        result_data = {
            "file_id": file_id,
            "name": uploaded_video.name,
            "frames": frame_number,
            "total_detections": total_detections,
            "elapsed": elapsed,
            "fps": fps,
            "width": width,
            "height": height,
            "detections": all_detections,
            "frame_counts": frame_counts,
            "key_frames": sorted(key_frames, key=lambda item: item[1]),
            "video_bytes": video_bytes,
        }

    except Exception as error:
        st.error("Video processing failed.")
        st.exception(error)

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        for temp_path in [temp_input, temp_output]:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except OSError:
                    pass

    return result_data


def video_workflow(prefix):
    state_key = f"{prefix}_video_result"

    uploaded_video = st.file_uploader(
        "Upload Road Video",
        type=["mp4", "avi", "mov", "mkv"],
        key=f"{prefix}_video_upload",
        help="Supported formats: MP4, AVI, MOV, MKV. Longer videos take longer to process."
    )

    if uploaded_video is None:
        empty_state(
            "🎬",
            "No video yet. Drop road footage above to start the analysis."
        )
        return

    if not model_status:
        st.warning("The model is not ready, so analysis is unavailable.")
        return

    file_id = (uploaded_video.name, uploaded_video.size)
    file_size_mb = uploaded_video.size / (1024 * 1024)

    render(
        f'<div class="file-chip">🎞️ <b>{escape(uploaded_video.name)}</b>'
        f'<span style="color:#8da2bf;">{file_size_mb:.1f} MB</span></div>'
    )

    clicked = st.button(
        "🎥 Start Video Analysis",
        use_container_width=True,
        key=f"{prefix}_video_start"
    )

    if clicked:
        data = process_video(uploaded_video, file_id)

        if data:
            dataframe = pd.DataFrame(data["detections"]) if data["detections"] else None
            csv_bytes = (
                dataframe.to_csv(index=False).encode("utf-8")
                if dataframe is not None else None
            )

            frames_with_damage = sum(1 for c in data["frame_counts"] if c > 0)

            pdf_bytes = safe_pdf(
                "Video Inspection Report",
                f"Frame-by-frame road damage analysis of {data['name']}",
                [
                    ("File", data["name"]),
                    ("Analysed on", datetime.now().strftime("%d %b %Y, %H:%M")),
                    ("Resolution", f"{data['width']} x {data['height']} px at {data['fps']:.1f} fps"),
                    ("Frames processed", str(data["frames"])),
                    ("Frames with damage", str(frames_with_damage)),
                    ("Processing time", f"{data['elapsed']:.1f} sec"),
                    ("Confidence threshold", f"{confidence:.2f}"),
                    ("Model", MODEL_PATH.name),
                ],
                summarize(data["detections"]),
                [
                    (f"Frame {number} ({count} detections)", jpeg, w, h)
                    for count, number, jpeg, w, h in data["key_frames"]
                ],
                dataframe
            )

            data["csv"] = csv_bytes
            data["pdf"] = pdf_bytes
            data["frames_with_damage"] = frames_with_damage

            st.session_state[state_key] = data
            add_history("Video", data["name"], data["detections"], pdf_bytes, csv_bytes)
            st.toast("Video analysis completed", icon="✅")

    res = st.session_state.get(state_key)

    if res and res["file_id"] == file_id:
        if clicked:
            st.success("Video analysis completed.")

        section_title("Video Analysis Summary")

        cards = st.columns(4)
        summary_cards = [
            ("Frames Processed", res["frames"], "Total frames analysed", "teal"),
            ("Total Frame Detections", res["total_detections"], "Across all frames", "blue"),
            ("Frames With Damage", res["frames_with_damage"], "At least one detection", "amber"),
            ("Processing Time", f"{res['elapsed']:.1f} sec", f"{res['width']} x {res['height']} px", "red"),
        ]
        for column, (label, value, hint, tone) in zip(cards, summary_cards):
            column.markdown(
                compact(metric_card(label, value, hint, tone)),
                unsafe_allow_html=True
            )

        if res["detections"]:
            show_metrics(res["detections"])

            chart1, chart2 = st.columns(2, gap="large")

            with chart1:
                render('<div class="panel-title">Detections per frame</div>')
                st.area_chart(
                    pd.DataFrame({"Detections": res["frame_counts"]}),
                    color="#14b8a6",
                    height=240
                )

            with chart2:
                render('<div class="panel-title">Damage by class</div>')
                show_class_chart(res["detections"])

            section_title("Detection Details")
            st.dataframe(
                pd.DataFrame(res["detections"]),
                use_container_width=True,
                hide_index=True,
                column_config=detection_table_config()
            )
        else:
            st.info("No road damage detected in the video.")

        section_title("Processed Video")

        st.video(res["video_bytes"])
        st.caption(
            "If the preview does not play in your browser, "
            "download the processed video and open it in a media player."
        )

        section_title("Download Reports")

        d1, d2, d3 = st.columns(3)

        with d1:
            pdf_download_button(
                res["pdf"], "roadvision_video_report.pdf", f"{prefix}_video_pdf"
            )

        with d2:
            if res["csv"]:
                st.download_button(
                    "⬇️ Download Video Detection Report",
                    data=res["csv"],
                    file_name="video_detection_report.csv",
                    mime="text/csv",
                    key=f"{prefix}_video_csv",
                    use_container_width=True
                )

        with d3:
            st.download_button(
                "⬇️ Download Processed Video",
                data=res["video_bytes"],
                file_name="roadvision_processed.mp4",
                mime="video/mp4",
                key=f"{prefix}_video_dl",
                use_container_width=True
            )


# -------------------------------------------------
# DASHBOARD PAGE
# -------------------------------------------------

if page == "Dashboard":

    render(f"""
    <div class="hero">
    <div class="hero-text">
    <h2>Spot road damage in seconds, from a photo or a drive-through video.</h2>
    <p>Upload road footage and RoadVision AI draws a box around every pothole and crack,
    estimates how severe each one looks, and gives you a PDF report to share.</p>
    <div class="chip-row">
    <span class="chip">YOLO detection</span>
    <span class="chip blue">Image and video</span>
    <span class="chip">PDF and CSV reports</span>
    <span class="chip blue">Annotated video export</span>
    </div>
    </div>
    <div class="hero-art">{HERO_SVG}</div>
    </div>
    """)

    if model_status:
        st.success("YOLO model loaded successfully.")
    else:
        st.error("Model is not ready. Check the model file and error above.")

    history = st.session_state["history"]
    image_runs = sum(1 for h in history if h["kind"] == "Image")
    video_runs = sum(1 for h in history if h["kind"] == "Video")
    total_found = sum(h["stats"]["total"] for h in history)
    high_found = sum(h["stats"]["high"] for h in history)

    kpis = st.columns(4)
    kpi_cards = [
        ("Images analysed", image_runs, "This session", "teal"),
        ("Videos analysed", video_runs, "This session", "blue"),
        ("Detections found", total_found, "Across all analyses", "amber"),
        ("High severity", high_found, "Large visible area", "red"),
    ]
    for column, (label, value, hint, tone) in zip(kpis, kpi_cards):
        column.markdown(
            compact(metric_card(label, value, hint, tone)),
            unsafe_allow_html=True
        )

    section_title("Quick Analysis")
    st.caption("Upload and analyse right here. No need to switch pages.")

    quick_image, quick_video = st.tabs(["🖼️ Analyse an image", "🎬 Analyse a video"])

    with quick_image:
        image_workflow("quick")

    with quick_video:
        video_workflow("quick")

    section_title("Go to")

    nav_cols = st.columns(4)
    shortcuts = [
        ("🖼️ Image Inspection", "Image Inspection"),
        ("🎬 Video Inspection", "Video Inspection"),
        ("📄 Reports", "Reports"),
        ("🧭 Road Guide", "Road Guide"),
    ]
    for column, (label, target) in zip(nav_cols, shortcuts):
        column.button(
            label,
            key=f"goto_{target}",
            on_click=go_to,
            args=(target,),
            use_container_width=True
        )

    section_title("Detection Classes")

    class_columns = st.columns(5)

    for column, name, icon in zip(class_columns, CLASS_NAMES, CLASS_ICONS):
        column.markdown(
            compact(
                f'<div class="class-card"><div class="class-icon">{icon}</div>'
                f'<div class="class-name">{name}</div>'
                f'<div class="class-tag">Road damage</div></div>'
            ),
            unsafe_allow_html=True
        )

    section_title("Session Activity")

    class_totals = {}
    for item in history:
        for name, count in item["stats"]["classes"].items():
            class_totals[name] = class_totals.get(name, 0) + count

    act1, act2 = st.columns([3, 2], gap="large")

    with act1:
        render('<div class="panel-title">Damage found by class</div>')
        if class_totals:
            st.bar_chart(
                pd.Series(class_totals).sort_values(ascending=False),
                color="#3b82f6",
                height=260
            )
        else:
            empty_state("📈", "Run an analysis and your results will chart here.")

    with act2:
        render('<div class="panel-title">How it works</div>')
        steps = [
            ("Upload", "Add a road image or a short video."),
            ("Detect", "The YOLO model finds and labels damage."),
            ("Review", "Check severity, charts and the annotated output."),
            ("Export", "Download a PDF, CSV or annotated video."),
        ]
        for index, (title, text) in enumerate(steps, start=1):
            render(
                f'<div class="step"><div class="step-num">{index}</div>'
                f'<div><div class="step-title">{title}</div>'
                f'<div class="step-text">{text}</div></div></div>'
            )

    st.markdown("---")
    st.info(
        "Severity is estimated from the visible bounding-box area. "
        "It does not measure actual pothole depth, physical dimensions "
        "or road safety risk."
    )


# -------------------------------------------------
# IMAGE INSPECTION PAGE
# -------------------------------------------------

elif page == "Image Inspection":

    page_header(
        "🖼️",
        "Image Inspection",
        "Upload an image to detect road damage"
    )

    image_workflow("page")

    with st.expander("Tips for better results"):
        st.markdown(
            "- Use a clear, well-lit photo taken from above or at a slight angle.\n"
            "- Keep the road surface in focus and avoid heavy shadows.\n"
            "- Lower the confidence slider if damage is missed, raise it if you see false detections.\n"
            "- Severity compares box size to image size, so the same pothole can score differently at different distances."
        )


# -------------------------------------------------
# VIDEO INSPECTION PAGE
# -------------------------------------------------

elif page == "Video Inspection":

    page_header(
        "🎬",
        "Video Inspection",
        "Analyze road videos frame by frame"
    )

    video_workflow("page")

    with st.expander("Tips for better results"):
        st.markdown(
            "- Use steady footage from a dashboard or handlebar camera.\n"
            "- Shorter clips process much faster, because every frame goes through the model.\n"
            "- Keep this tab open while a video is processing.\n"
            "- Never film while driving. Have a passenger record or mount the camera before you set off."
        )


# -------------------------------------------------
# REPORTS PAGE
# -------------------------------------------------

elif page == "Reports":

    page_header(
        "📄",
        "Reports",
        "Download PDF and CSV reports from your analyses in this session"
    )

    history = st.session_state["history"]

    if not history:
        empty_state(
            "📄",
            "No reports yet. Analyse an image or video and its report will be listed here."
        )
        st.button(
            "Go to Dashboard",
            on_click=go_to,
            args=("Dashboard",),
            key="reports_goto_dash"
        )
    else:
        if not REPORTLAB_OK:
            st.info("PDF reports need ReportLab. Run: pip install reportlab")

        for index, item in enumerate(history):
            stats = item["stats"]
            icon = "🖼️" if item["kind"] == "Image" else "🎬"

            info_col, pdf_col, csv_col = st.columns([4, 1.4, 1.4], gap="medium")

            with info_col:
                render(
                    f'<div class="hist-card"><div class="hist-title">{icon} {escape(item["name"])}</div>'
                    f'<div class="hist-meta">{item["kind"]} analysis | {item["time"]} | '
                    f'{stats["total"]} detections | {stats["potholes"]} potholes | '
                    f'High {stats["high"]}, Medium {stats["medium"]}, Low {stats["low"]}</div></div>'
                )

            with pdf_col:
                if item["pdf"]:
                    st.download_button(
                        "📄 PDF",
                        data=item["pdf"],
                        file_name=f"roadvision_{item['kind'].lower()}_report_{index + 1}.pdf",
                        mime="application/pdf",
                        key=f"hist_pdf_{index}",
                        use_container_width=True
                    )

            with csv_col:
                if item["csv"]:
                    st.download_button(
                        "⬇️ CSV",
                        data=item["csv"],
                        file_name=f"roadvision_{item['kind'].lower()}_report_{index + 1}.csv",
                        mime="text/csv",
                        key=f"hist_csv_{index}",
                        use_container_width=True
                    )

        st.caption("Reports are kept for this browser session only. Download what you need before closing the tab.")

        st.button(
            "Clear report history",
            key="clear_history",
            on_click=lambda: st.session_state.update({"history": []})
        )


# -------------------------------------------------
# ROAD GUIDE PAGE
# -------------------------------------------------

elif page == "Road Guide":

    page_header(
        "🧭",
        "Road Damage Guide",
        "Annotated diagrams of the damage types this model looks for"
    )

    render(
        '<div class="info-card"><h3>How to read the diagrams</h3>'
        '<p>Each diagram shows a road from above, with lane markings. The dashed teal box and '
        'label mimic what the detector draws on your images. Orange lines mark the damage itself.</p></div>'
    )

    section_title("Damage Types")

    for start in range(0, len(DAMAGE_GUIDE), 3):
        row = DAMAGE_GUIDE[start:start + 3]
        cols = st.columns(3, gap="medium")
        for col, item in zip(cols, row):
            col.markdown(
                compact(
                    f'<div class="guide-card">'
                    f'{damage_svg(item["kind"], item["name"], item["note"])}'
                    f'<h4>{item["name"]}</h4>'
                    f'<p><b>What it is:</b> {item["what"]}</p>'
                    f'<p><b>Common cause:</b> {item["cause"]}</p>'
                    f'<p><b>Typical response:</b> {item["response"]}</p></div>'
                ),
                unsafe_allow_html=True
            )

    section_title("Why Potholes Form")

    cross_col, text_col = st.columns([3, 2], gap="large")

    with cross_col:
        render(f'<div class="svg-card">{CROSS_SECTION_SVG}</div>')

    with text_col:
        render(
            '<div class="info-card"><h3>Cracks let water in</h3>'
            '<p>A road is built in layers. Once a crack opens in the surface, rain gets through and '
            'softens the layers below. Traffic then pushes the weakened material out, and a pothole forms. '
            'Sealing cracks early is far cheaper than repairing potholes later.</p></div>'
        )

    section_title("How Severity Is Estimated")

    sev_cols = st.columns(3)
    severity_cards = [
        ("Low", "Under 2.5% of the image", "Small in the frame. May be minor or far away.", "teal"),
        ("Medium", "2.5% to 8% of the image", "Clearly visible. Worth noting for maintenance.", "amber"),
        ("High", "8% or more of the image", "Large in the frame. Check on site as a priority.", "red"),
    ]
    for column, (name, rule, text, tone) in zip(sev_cols, severity_cards):
        column.markdown(
            compact(metric_card(f"{name} severity", name, f"{rule}. {text}", tone)),
            unsafe_allow_html=True
        )

    st.info(
        "These levels come from box size only. A small pothole filmed up close can look 'High', "
        "and a deep one filmed from far away can look 'Low'."
    )


# -------------------------------------------------
# SAFETY PAGE
# -------------------------------------------------

elif page == "Safety":

    page_header(
        "🛡️",
        "Safety, Disclaimer and Official Links",
        "Please read before relying on any result"
    )

    render(
        '<div class="info-card danger"><h3>⚠️ Important disclaimer</h3>'
        '<ul>'
        '<li>RoadVision AI is a student research and learning project. Its results are automated estimates '
        'and can miss real damage or flag damage that is not there.</li>'
        '<li>It does <b>not</b> predict, prevent or assess the risk of road accidents, and it cannot measure '
        'pothole depth or structural condition.</li>'
        '<li>Do not use it as the only basis for safety, maintenance, insurance or legal decisions. '
        'Damaged roads should be inspected by a qualified engineer or the responsible authority.</li>'
        '<li>The authors accept no liability for accidents, injury or loss arising from use of this app or its reports.</li>'
        '<li>Never use this app, or film road footage, while driving or riding.</li>'
        '</ul></div>'
    )

    section_title("Emergency Numbers (India)")

    render(
        '<div class="sos">'
        '<div><b>112</b>National emergency number</div>'
        '<div><b>108</b>Ambulance (most states)</div>'
        '<div><b>1033</b>NHAI highway helpline</div>'
        '</div>'
    )
    st.caption("Numbers can change. Check the official sites below if a number does not connect.")

    section_title("Driving Near Damaged Roads")

    tip1, tip2 = st.columns(2, gap="large")

    with tip1:
        render(
            '<div class="info-card warn"><h3>Stay safe on the road</h3><ul>'
            '<li>Slow down early instead of braking hard at the last moment.</li>'
            '<li>After rain, treat every puddle as a possible pothole. Water hides depth.</li>'
            '<li>Do not swerve suddenly. Check your mirrors first.</li>'
            '<li>Two-wheeler riders should take extra care, since a single pothole can cause a fall.</li>'
            '<li>At night, reduce speed and use full headlights where allowed.</li>'
            '</ul></div>'
        )

    with tip2:
        render(
            '<div class="info-card"><h3>Reporting road damage</h3><ul>'
            '<li>Note the exact location, such as the road name, landmark or GPS pin.</li>'
            '<li>Take a clear photo from a safe place on the roadside.</li>'
            '<li>Report national highway issues through NHAI and local roads through your municipal body or PWD.</li>'
            '<li>You can also lodge a complaint on the CPGRAMS grievance portal.</li>'
            '<li>Keep the complaint number so you can follow up.</li>'
            '</ul></div>'
        )

    section_title("Official and Useful Links")
    link_cards(OFFICIAL_LINKS, columns=4)
    st.caption("Links open external websites that are not run by this project. Check each site for the latest information.")


# -------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------

elif page == "About":

    page_header(
        "ℹ️",
        "About RoadVision AI",
        "What this project is and how it was built"
    )

    about_left, about_right = st.columns([3, 2], gap="large")

    with about_left:
        render(
            '<div class="info-card"><h3>What I built</h3>'
            '<p>RoadVision AI is a web application that finds road damage in photos and videos. '
            'A YOLO object-detection model locates potholes and four types of cracks, draws boxes around them, '
            'and the app estimates how severe each one looks, counts the results, charts them, '
            'and exports CSV, PDF and annotated-video reports. It is designed to make road inspection '
            'faster and easier to document.</p></div>'
        )

    with about_right:
        render(
            f'<div class="info-card"><h3>Project details</h3><ul>'
            f'<li><b>Author:</b> {escape(PROJECT_INFO["author"])}</li>'
            f'<li><b>Institute:</b> {escape(PROJECT_INFO["institute"])}</li>'
            f'<li><b>Type:</b> {escape(PROJECT_INFO["course"])}</li>'
            f'<li><b>Model file:</b> {MODEL_PATH.name}</li>'
            f'<li><b>Classes:</b> {len(CLASS_NAMES)}</li></ul></div>'
        )

    section_title("How It Works")

    render(
        '<div class="flow">'
        '<div class="flow-node"><div class="fi">📤</div><div class="ft">Upload</div><div class="fd">Image or video</div></div>'
        '<div class="flow-node"><div class="fi">🧠</div><div class="ft">Detect</div><div class="fd">YOLO finds damage</div></div>'
        '<div class="flow-node"><div class="fi">📐</div><div class="ft">Estimate</div><div class="fd">Severity from box area</div></div>'
        '<div class="flow-node"><div class="fi">📊</div><div class="ft">Summarise</div><div class="fd">Metrics and charts</div></div>'
        '<div class="flow-node"><div class="fi">📄</div><div class="ft">Export</div><div class="fd">PDF, CSV, video</div></div>'
        '</div>'
    )

    section_title("Features")

    feat1, feat2 = st.columns(2, gap="large")

    with feat1:
        render(
            '<div class="info-card"><ul>'
            '<li>Image and video damage detection with adjustable confidence</li>'
            '<li>Quick analysis directly from the dashboard</li>'
            '<li>Severity estimate and distribution for every analysis</li>'
            '<li>Charts for damage by class and detections per frame</li>'
            '</ul></div>'
        )

    with feat2:
        render(
            '<div class="info-card"><ul>'
            '<li>PDF reports for both image and video results</li>'
            '<li>CSV exports and annotated video download</li>'
            '<li>Session report history</li>'
            '<li>Road damage guide, safety advice and official links</li>'
            '</ul></div>'
        )

    section_title("Technology")

    render(
        '<div class="chip-row">'
        '<span class="chip">Python</span><span class="chip blue">Streamlit</span>'
        '<span class="chip">Ultralytics YOLO</span><span class="chip blue">OpenCV</span>'
        '<span class="chip">Pandas</span><span class="chip blue">Pillow</span>'
        '<span class="chip">ReportLab</span></div>'
    )

    section_title("Limitations")

    render(
        '<div class="info-card warn"><ul>'
        '<li>Severity uses bounding-box area only, so it is not a measure of depth or danger.</li>'
        '<li>Accuracy depends on image quality, lighting, camera angle and the training data.</li>'
        '<li>Video is processed on the server frame by frame, so long clips take time.</li>'
        '<li>Results and reports exist for the current browser session only.</li>'
        '</ul></div>'
    )

    section_title("Learn More")

    about_links = [
        ("🤖", "Ultralytics YOLO", "Documentation for the detection framework.", "https://docs.ultralytics.com"),
        ("🎈", "Streamlit", "The framework used to build this interface.", "https://docs.streamlit.io"),
        ("📷", "OpenCV", "Library used for reading and writing video frames.", "https://docs.opencv.org"),
    ]
    if PROJECT_INFO["github"]:
        about_links.append(("💻", "Project source code", "See how this app was built.", PROJECT_INFO["github"]))
    link_cards(about_links, columns=4)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

render_footer()