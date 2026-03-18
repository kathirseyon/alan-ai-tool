"""
=============================================================
  AI SALES PERSONALIZER — Alan Internship Project
  Author: Seyon Kathir | Built with Streamlit + OpenAI API
  
  Scrapes a company website and generates lead intelligence
  and a personalized outbound sales email using GPT.
=============================================================
"""

import os
import re
import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

# Load from .env file if it exists
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
# Paste your key here as a fallback, or set it in a .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# gpt-4o-mini is fast and cheap (~$0.001/run). Swap for "gpt-4o" for better quality.
MODEL = "gpt-4o-mini"

# Cap scraped text so the prompt stays focused
MAX_WEBSITE_CHARS = 3000

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sales Personalizer · Alan",
    page_icon="🐻",   # Blue bear — Alan's actual mascot
    layout="centered",
)

# ── CSS — Alan's real brand palette ───────────────────────────────────────────
# Alan Canada uses: deep navy #0f1733, purple/indigo #4f46e5, soft lavender bg
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark navy page background ── */
    .stApp {
        background-color: #0f1733;
    }

    /* ── Input fields: dark navy with light border ── */
    input[type="text"], [data-testid="stTextInput"] input {
        background-color: #1a2147 !important;
        color: #e8eaf6 !important;
        border: 1.5px solid #3d3f6e !important;
        border-radius: 8px !important;
        caret-color: #818cf8 !important;
    }
    input[type="text"]:focus, [data-testid="stTextInput"] input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.2) !important;
    }
    input::placeholder {
        color: #6366a0 !important;
    }

    /* ── Dropdown ── */
    [data-testid="stSelectbox"] > div > div {
        background-color: #1a2147 !important;
        color: #e8eaf6 !important;
        border: 1.5px solid #3d3f6e !important;
        border-radius: 8px !important;
    }

    /* ── Form container ── */
    [data-testid="stForm"] {
        background-color: #151c3d;
        border: 1px solid #2a2f5e;
        border-radius: 16px;
        padding: 1.5rem;
    }

    /* ── Labels above inputs ── */
    label, [data-testid="stWidgetLabel"] p {
        color: #c7c9e8 !important;
        font-weight: 500 !important;
    }

    /* ── Hero header ── */
    .alan-header {
        background: linear-gradient(135deg, #1e2452 0%, #2d1f6e 100%);
        border: 1px solid #3b3480;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .alan-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: white;
    }
    .alan-header p {
        color: #a5b0f5;
        margin: 0;
        font-size: 0.97rem;
    }
    .alan-dot {
        display: inline-block;
        width: 11px;
        height: 11px;
        background: #818cf8;
        border-radius: 50%;
        margin-right: 9px;
        vertical-align: middle;
    }
    .alan-dot2 {
        display: inline-block;
        width: 7px;
        height: 7px;
        background: #4f46e5;
        border-radius: 50%;
        margin-right: 3px;
        vertical-align: middle;
        position: relative;
        top: -1px;
    }

    /* ── Submit button ── */
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #4f46e5, #6d63f5) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.5rem !important;
        font-size: 0.97rem !important;
        transition: opacity 0.2s;
    }
    div.stFormSubmitButton > button:hover {
        opacity: 0.88 !important;
    }

    /* ── Section title ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c7c9e8;
        margin: 1.5rem 0 0.75rem 0;
        letter-spacing: 0.01em;
    }

    /* ── Result cards ── */
    .result-card {
        background: #151c3d;
        border: 1px solid #2a2f5e;
        border-left: 4px solid #4f46e5;
        border-radius: 0 12px 12px 0;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1rem;
    }
    .result-card .card-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #818cf8;
        margin-bottom: 0.35rem;
    }
    .result-card .card-value {
        font-size: 0.95rem;
        color: #d4d6f0;
        line-height: 1.65;
    }

    /* ── Lead score badges ── */
    .badge-high   { background:#14532d; color:#86efac; padding:3px 12px; border-radius:99px; font-weight:600; font-size:0.82rem; }
    .badge-medium { background:#78350f; color:#fcd34d; padding:3px 12px; border-radius:99px; font-weight:600; font-size:0.82rem; }
    .badge-low    { background:#7f1d1d; color:#fca5a5; padding:3px 12px; border-radius:99px; font-weight:600; font-size:0.82rem; }

    /* ── Email display box ── */
    .email-box {
        background: #151c3d;
        border: 1px solid #2a2f5e;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        white-space: pre-wrap;
        font-size: 0.93rem;
        line-height: 1.8;
        color: #d4d6f0;
        margin-bottom: 0.75rem;
    }

    /* ── Copy button ── */
    .copy-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #1e2452;
        border: 1.5px solid #4f46e5;
        color: #818cf8;
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        font-weight: 600;
        padding: 7px 18px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.18s;
        margin-bottom: 1rem;
    }
    .copy-btn:hover {
        background: #4f46e5;
        color: white;
    }
    .copy-btn.copied {
        background: #14532d;
        border-color: #22c55e;
        color: #86efac;
    }

    /* ── Footer text ── */
    footer, [data-testid="stDecoration"] { display: none; }
    .footer-text {
        color: #3d3f6e;
        font-size: 0.8rem;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }

    /* ── Divider ── */
    hr {
        border-color: #2a2f5e !important;
    }

    /* ── Status/spinner box ── */
    [data-testid="stStatusWidget"] {
        background: #151c3d !important;
        border: 1.5px solid #4f46e5 !important;
        border-radius: 10px !important;
        color: #c7c9e8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="alan-header">
    <h1>
        <span class="alan-dot"></span><span class="alan-dot2"></span>
        AI Sales Personalizer
    </h1>
    <p>Enter a company's website to instantly generate lead intelligence and a personalized outbound email — powered by GPT.</p>
</div>
""", unsafe_allow_html=True)


# ── Helper: scrape website ─────────────────────────────────────────────────────
def scrape_website(url: str) -> str:
    """
    Fetches a company homepage and returns cleaned, readable text.
    Removes nav/scripts/footers and truncates to MAX_WEBSITE_CHARS
    so the AI prompt stays focused and cheap to run.
    """
    headers = {
        # Mimic a real browser to avoid being blocked by anti-bot checks
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # throws on 4xx/5xx errors
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Could not fetch the website: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Strip tags that contribute noise but no real content
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())  # collapse all whitespace into single spaces
    return text[:MAX_WEBSITE_CHARS]


# ── Helper: extract a labeled section from AI output ──────────────────────────
def extract_section(text: str, label: str) -> str:
    """
    Finds 'Label: ...' in the AI response and returns the content
    up until the next labeled section (or end of string).
    """
    pattern = rf"{re.escape(label)}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ── Helper: call OpenAI ────────────────────────────────────────────────────────
def generate_insights(company_name: str, target_role: str, website_text: str) -> dict:
    """
    Sends scraped website text to GPT with a structured prompt.
    Returns a dict: summary, size, score, reason, email, raw.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "Get a key at https://platform.openai.com/api-keys\n"
            "and paste it into line 40 of app.py."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Each section uses a consistent 'Label:' format so extract_section() can parse it
    prompt = f"""
You are an expert B2B sales analyst helping a company called Alan.
Alan sells employee benefits and health insurance to Canadian companies with 3–500 employees.

Below is scraped text from the homepage of {company_name}.

WEBSITE TEXT:
\"\"\"
{website_text}
\"\"\"

Produce the following 5 sections for a sales rep targeting the {target_role} at {company_name}.
Use EXACTLY these labels (with a colon) so the output can be parsed programmatically:

Company Summary:
2 sentences explaining what {company_name} does.

Estimated Company Size:
Best estimate of employee count from signals in the text (e.g. "~50–100 employees").

Lead Score:
One word only — High, Medium, or Low — based on fit with Alan's target (3–500 employee Canadian tech companies).

Reason:
1–2 sentences explaining the lead score.

Personalized Outreach Email:
A warm, human cold email under 150 words from Alan's sales team to the {target_role} at {company_name}.
Reference something specific about the company. End with a soft CTA like a 15-min call.
"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content

    return {
        "summary": extract_section(raw, "Company Summary"),
        "size":    extract_section(raw, "Estimated Company Size"),
        "score":   extract_section(raw, "Lead Score").strip().capitalize(),
        "reason":  extract_section(raw, "Reason"),
        "email":   extract_section(raw, "Personalized Outreach Email"),
        "raw":     raw,  # fallback if parsing misses a section
    }


# ── UI: Input form ─────────────────────────────────────────────────────────────
with st.form("sales_form"):
    col1, col2 = st.columns(2)

    with col1:
        company_name = st.text_input("🏢 Company Name", placeholder="e.g. Shopify")

    with col2:
        target_role = st.selectbox(
            "🎯 Target Role",
            ["HR Leader", "Head of People", "CFO", "Founder"],
        )

    company_url = st.text_input(
        "🌐 Company Website URL",
        placeholder="https://www.example.com",
    )

    submitted = st.form_submit_button("✨ Generate Insights & Email", use_container_width=True)


# ── Processing & Output ────────────────────────────────────────────────────────
if submitted:
    if not company_name.strip():
        st.warning("Please enter a company name.")
    elif not company_url.strip():
        st.warning("Please enter a company website URL.")
    else:
        # Auto-prepend https:// if missing
        url = company_url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        # Step 1 — Scrape
        with st.status("🔍 Step 1 of 2 — Scraping website...", expanded=True) as s1:
            try:
                website_text = scrape_website(url)
                s1.update(label=f"✅ Scraped {len(website_text)} chars from {url}", state="complete")
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        # Step 2 — Generate insights
        with st.status("🤖 Step 2 of 2 — Generating insights with GPT...", expanded=True) as s2:
            try:
                result = generate_insights(company_name.strip(), target_role, website_text)
                s2.update(label="✅ Done! Insights ready.", state="complete")
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Lead Intelligence</div>', unsafe_allow_html=True)

        # Summary + Size
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-label">Company Summary</div>
                <div class="card-value">{result['summary'] or "—"}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-label">Estimated Size</div>
                <div class="card-value">{result['size'] or "—"}</div>
            </div>""", unsafe_allow_html=True)

        # Lead score + reason
        score = result['score']
        badge_class = (
            "badge-high"   if "high"   in score.lower() else
            "badge-medium" if "medium" in score.lower() else
            "badge-low"
        )
        st.markdown(f"""
        <div class="result-card">
            <div class="card-label">Lead Score &nbsp;<span class="{badge_class}">{score or "—"}</span></div>
            <div class="card-value">{result['reason'] or "—"}</div>
        </div>""", unsafe_allow_html=True)

        # ── Email ─────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">✉️ Personalized Outreach Email</div>', unsafe_allow_html=True)

        email_text = result['email'] or result['raw']

        # Display the email in a styled box
        st.markdown(f'<div class="email-box">{email_text}</div>', unsafe_allow_html=True)

        # JavaScript copy button — copies email_text to clipboard on click
        # We embed the email text safely into a JS string using repr()
        safe_email = email_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        st.components.v1.html(f"""
        <button class="copy-btn" id="copyBtn" onclick="
            navigator.clipboard.writeText(`{safe_email}`).then(() => {{
                const btn = document.getElementById('copyBtn');
                btn.textContent = '✅  Copied!';
                btn.classList.add('copied');
                setTimeout(() => {{
                    btn.textContent = '📋  Copy Email';
                    btn.classList.remove('copied');
                }}, 2500);
            }});
        ">📋  Copy Email</button>
        <style>
            .copy-btn {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: #1e2452;
                border: 1.5px solid #4f46e5;
                color: #818cf8;
                font-family: 'Inter', sans-serif;
                font-size: 0.88rem;
                font-weight: 600;
                padding: 8px 20px;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.18s;
            }}
            .copy-btn:hover {{ background: #4f46e5; color: white; }}
            .copy-btn.copied {{ background: #14532d; border-color: #22c55e; color: #86efac; }}
        </style>
        """, height=50)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer-text">Built for Alan · AI Sales Personalizer · Powered by OpenAI GPT · By Seyon Kathir</div>', unsafe_allow_html=True)
