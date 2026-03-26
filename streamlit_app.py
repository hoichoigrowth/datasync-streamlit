"""
DataSync Pro – Streamlit App
PI Data Processing for Targeting
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from collections import defaultdict

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="DataSync Pro", page_icon="📊", layout="wide")

# ── Reference data ───────────────────────────────────────────────
COUNTRY_INFO = {
    "1":   {"name": "USA/Canada",      "length": 10},
    "44":  {"name": "United Kingdom",   "length": 10},
    "60":  {"name": "Malaysia",         "length": 9},
    "65":  {"name": "Singapore",        "length": 8},
    "971": {"name": "UAE",              "length": 9},
    "966": {"name": "Saudi Arabia",     "length": 9},
    "61":  {"name": "Australia",        "length": 9},
    "91":  {"name": "India",            "length": 10},
    "880": {"name": "Bangladesh",       "length": 8},
}

CURRENCY_MAP = {
    "USD": {"code": "1",   "length": 10, "name": "US Dollar"},
    "GBP": {"code": "44",  "length": 10, "name": "British Pound"},
    "MYR": {"code": "60",  "length": 9,  "name": "Malaysian Ringgit"},
    "SGD": {"code": "65",  "length": 8,  "name": "Singapore Dollar"},
    "AED": {"code": "971", "length": 9,  "name": "UAE Dirham"},
    "SAR": {"code": "966", "length": 9,  "name": "Saudi Riyal"},
    "AUD": {"code": "61",  "length": 9,  "name": "Australian Dollar"},
    "CAD": {"code": "1",   "length": 10, "name": "Canadian Dollar"},
    "INR": {"code": "91",  "length": 10, "name": "Indian Rupee"},
    "BDT": {"code": "880", "length": 8,  "name": "Bangladeshi Taka"},
}

ALL_MARKETS = list(CURRENCY_MAP.keys())

# ── Helpers ──────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
DIGITS_RE = re.compile(r"\D")

# Patterns to find the header row
HEADER_PATTERNS = [
    re.compile(r"^e[\-_\s]?mail", re.I),
    re.compile(r"^phone", re.I),
    re.compile(r"^mobile", re.I),
    re.compile(r"^currency", re.I),
]

# Normalize column names
COLUMN_NORMALIZE = [
    (re.compile(r"^e[\-_\s]?mail[\s_\-]*(address|id)?$", re.I), "Email"),
    (re.compile(r"^(phone|mobile|cell|contact)[\s_\-]*(number|no|num)?$", re.I), "Phone"),
    (re.compile(r"^currency[\s_\-]*(code)?$", re.I), "currency"),
]


def normalize_col(name):
    s = str(name).strip()
    for pat, norm in COLUMN_NORMALIZE:
        if pat.match(s):
            return norm
    return s


def is_valid_email(e):
    return bool(e) and EMAIL_RE.match(str(e).strip()) is not None


def is_internal(e):
    lo = str(e).lower()
    return lo.endswith("@hoichoi.tv") or lo.endswith("@hoichoitv.com")


def digits_only(s):
    return DIGITS_RE.sub("", str(s) if pd.notna(s) else "")


def find_header_row(df_raw):
    """Scan first 10 rows for a row that looks like headers."""
    limit = min(len(df_raw), 10)
    for i in range(limit):
        row_vals = [str(v).strip() for v in df_raw.iloc[i]]
        matches = sum(1 for v in row_vals if any(p.match(v) for p in HEADER_PATTERNS))
        if matches >= 1:
            return i
    return 0


def phone_to_int_str(val):
    """Convert phone value to integer string (0 decimals). Handles scientific notation."""
    if pd.isna(val) or str(val).strip() == "":
        return ""
    s = str(val).strip()
    try:
        n = float(s)
        if np.isnan(n) or np.isinf(n):
            return s
        return str(int(round(n)))
    except (ValueError, TypeError):
        return s


# ── File parser ──────────────────────────────────────────────────
def parse_file(uploaded_file):
    """
    Parse uploaded CSV/XLSX:
    1. Auto-detect header row
    2. Normalize column names
    3. Convert Phone to integer strings
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    # Read as raw (no header)
    if name.endswith(".csv"):
        # Try encodings in order, skip bad lines, use python engine as fallback
        df_raw = None
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                df_raw = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    header=None,
                    dtype=str,
                    keep_default_na=False,
                    encoding=enc,
                    on_bad_lines="skip",
                    engine="python",
                )
                break
            except Exception:
                continue
        if df_raw is None:
            return pd.DataFrame(), "Could not parse CSV — try saving as UTF-8"
    else:
        try:
            df_raw = pd.read_excel(io.BytesIO(raw_bytes), header=None, dtype=str, engine="openpyxl")
        except Exception:
            df_raw = pd.read_excel(io.BytesIO(raw_bytes), header=None, dtype=str, engine="xlrd")
        df_raw = df_raw.fillna("")

    if df_raw.empty:
        return pd.DataFrame(), "File is empty"

    # Find header row
    header_idx = find_header_row(df_raw)

    # Use that row as headers
    headers = [normalize_col(v) for v in df_raw.iloc[header_idx]]
    df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)
    df.columns = headers

    # Drop fully empty rows
    df = df.dropna(how="all")
    df = df[df.apply(lambda row: any(str(v).strip() != "" for v in row), axis=1)]

    # Convert Phone to integer string
    if "Phone" in df.columns:
        df["Phone"] = df["Phone"].apply(phone_to_int_str)

    info = f"Header found at row {header_idx + 1}. Columns: {', '.join(headers)}. {len(df)} data rows."
    return df, info


# ── Processing logic ─────────────────────────────────────────────
def process_internal(rows):
    results = []
    sorted_codes = sorted(COUNTRY_INFO.items(), key=lambda x: -len(x[0]))
    for _, row in rows.iterrows():
        email = str(row.get("Email", "")).strip()
        if not is_valid_email(email) or not is_internal(email):
            continue
        phone_part = email.split("@")[0].strip()
        formatted = phone_part
        if phone_part.startswith("+"):
            for code, info in sorted_codes:
                if phone_part[1:].startswith(code):
                    number_part = digits_only(phone_part[1 + len(code):])
                    if number_part:
                        formatted = f"{code} {number_part[-info['length']:]}"
                    break
        results.append({"Email": email, "Phone": formatted})
    return results


def process_external(rows, markets):
    results = []
    market_set = set(markets) if markets else None
    for _, row in rows.iterrows():
        email = str(row.get("Email", "")).strip()
        if not is_valid_email(email) or is_internal(email):
            continue
        raw_phone = digits_only(row.get("Phone", ""))
        if len(raw_phone) < 5:
            continue
        currency = str(row.get("currency", "")).strip()
        if market_set and currency not in market_set:
            continue
        mapping = CURRENCY_MAP.get(currency)
        if not mapping:
            continue
        phone = raw_phone
        if phone.startswith(mapping["code"]):
            phone = phone[len(mapping["code"]):]
        formatted = f"{mapping['code']} {phone[:mapping['length']]}"
        results.append({"Email": email, "Phone": formatted})
    return results


def process_data(df, markets, lowercase_emails=True):
    if df.empty:
        return pd.DataFrame(), {}, [], []

    if lowercase_emails and "Email" in df.columns:
        df = df.copy()
        df["Email"] = df["Email"].str.lower()

    internal = process_internal(df)
    external = process_external(df, markets)
    all_rows = internal + external

    # Deduplicate
    seen = set()
    deduped = []
    for r in all_rows:
        key = f"{r['Email']}|{r['Phone']}"
        if key in seen or not r.get("Phone", "").strip():
            continue
        seen.add(key)
        deduped.append(r)

    # Group by email
    grouped = defaultdict(list)
    for r in deduped:
        grouped[r["Email"]].append(r["Phone"])

    final = [{"Email": em, "Phone": ", ".join(phs)} for em, phs in grouped.items()]
    final_df = pd.DataFrame(final) if final else pd.DataFrame(columns=["Email", "Phone"])

    # Stats
    total = len(df)
    valid_emails = df["Email"].apply(lambda e: is_valid_email(str(e).strip())).sum() if "Email" in df.columns else 0
    valid_phones = df["Phone"].apply(lambda p: len(digits_only(p)) >= 5).sum() if "Phone" in df.columns else 0
    internal_count = df["Email"].apply(lambda e: is_internal(str(e).strip())).sum() if "Email" in df.columns else 0

    def pct(n, d):
        return round((n / d) * 100, 1) if d else 0

    stats = {
        "Total Rows": total,
        "Valid Emails": f"{valid_emails} ({pct(valid_emails, total)}%)",
        "Valid Phones": f"{valid_phones} ({pct(valid_phones, total)}%)",
        "Internal Emails": int(internal_count),
        "Processed Records": len(final_df),
    }

    # Country split
    country_split = defaultdict(int)
    for r in deduped:
        code = r["Phone"].split(" ")[0]
        name = COUNTRY_INFO.get(code, {}).get("name", "Unknown")
        country_split[name] += 1

    # Currency split
    currency_split = defaultdict(int)
    if "currency" in df.columns:
        for val in df["currency"]:
            c = str(val).strip() or "Unknown"
            currency_split[c] += 1

    return final_df, stats, dict(country_split), dict(currency_split)


# ── UI ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .stat-card { background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stat-value { font-size: 28px; font-weight: 700; }
    .stat-label { font-size: 13px; color: #6b7280; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 📊 DataSync Pro")
st.markdown("**PI Data Processing for Targeting** — Upload your data, process phone numbers by market, and download clean results.")
st.divider()

# Sidebar options
with st.sidebar:
    st.header("⚙️ Processing Options")
    selected_markets = st.multiselect("Select Markets", ALL_MARKETS, default=["INR", "BDT", "USD"])
    lowercase_emails = st.checkbox("Lowercase emails", value=True)
    st.divider()
    st.caption("DataSync Pro v2.0 • Streamlit Edition")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    with st.spinner("Parsing file..."):
        df, parse_info = parse_file(uploaded_file)

    st.success(f"**{uploaded_file.name}** — {parse_info}")

    if not df.empty:
        # Preview
        with st.expander(f"📋 Preview (first 10 of {len(df)} rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        # Process button
        if st.button("🚀 Process Data", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                final_df, stats, country_split, currency_split = process_data(
                    df, selected_markets, lowercase_emails
                )

            st.session_state["result"] = final_df
            st.session_state["stats"] = stats
            st.session_state["country_split"] = country_split
            st.session_state["currency_split"] = currency_split

    # Show results if they exist
    if "result" in st.session_state and st.session_state["result"] is not None:
        final_df = st.session_state["result"]
        stats = st.session_state["stats"]
        country_split = st.session_state["country_split"]
        currency_split = st.session_state["currency_split"]

        st.divider()
        st.subheader("📈 Results")

        # Stat cards
        cols = st.columns(5)
        for i, (label, value) in enumerate(stats.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🌍 Country Split**")
            if country_split:
                chart_df = pd.DataFrame(list(country_split.items()), columns=["Country", "Count"])
                chart_df = chart_df.sort_values("Count", ascending=False)
                st.bar_chart(chart_df.set_index("Country"))
            else:
                st.info("No country data")

        with col2:
            st.markdown("**💱 Currency Distribution**")
            if currency_split:
                chart_df = pd.DataFrame(list(currency_split.items()), columns=["Currency", "Count"])
                chart_df = chart_df.sort_values("Count", ascending=False)
                st.bar_chart(chart_df.set_index("Currency"))
            else:
                st.info("No currency data")

        # Processed data table
        st.markdown(f"**Processed Data ({len(final_df)} records)**")
        st.dataframe(final_df, use_container_width=True, height=400)

        # Downloads
        col_dl1, col_dl2, _ = st.columns([1, 1, 3])
        with col_dl1:
            csv_data = final_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv_data, "processed_data.csv", "text/csv", use_container_width=True)
        with col_dl2:
            buf = io.BytesIO()
            final_df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button("📥 Download XLSX", buf.getvalue(), "processed_data.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
    else:
        st.info("👆 Upload a file and click **Process Data** to see results.")
