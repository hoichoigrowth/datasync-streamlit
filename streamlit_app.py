"""
DataSync Pro — PI Data Processing for Targeting
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from collections import defaultdict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataSync Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .section-title {font-size: 14px; font-weight: 600; color: #374151; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;}
    .stDataFrame {border-radius: 8px; overflow: hidden;}
    div[data-testid="stHorizontalBlock"] > div {gap: 1rem;}
</style>
""", unsafe_allow_html=True)

# ── Reference data ────────────────────────────────────────────────────────────
COUNTRY_INFO = {
    "1":   {"name": "USA/Canada",     "length": 10},
    "44":  {"name": "UK",             "length": 10},
    "60":  {"name": "Malaysia",       "length": 9},
    "65":  {"name": "Singapore",      "length": 8},
    "971": {"name": "UAE",            "length": 9},
    "966": {"name": "Saudi Arabia",   "length": 9},
    "61":  {"name": "Australia",      "length": 9},
    "91":  {"name": "India",          "length": 10},
    "880": {"name": "Bangladesh",     "length": 8},
}

CURRENCY_MAP = {
    "USD": {"code": "1",   "length": 10},
    "GBP": {"code": "44",  "length": 10},
    "MYR": {"code": "60",  "length": 9},
    "SGD": {"code": "65",  "length": 8},
    "AED": {"code": "971", "length": 9},
    "SAR": {"code": "966", "length": 9},
    "AUD": {"code": "61",  "length": 9},
    "CAD": {"code": "1",   "length": 10},
    "INR": {"code": "91",  "length": 10},
    "BDT": {"code": "880", "length": 8},
}

# ── Helpers ───────────────────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
DIGITS_RE = re.compile(r"\D")

COLUMN_NORMALIZE = [
    (re.compile(r"^e[\-_\s]?mail.*$", re.I), "Email"),
    (re.compile(r"^(phone|mobile|cell|contact).*$", re.I), "Phone"),
    (re.compile(r"^currency.*$", re.I), "currency"),
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

def phone_to_int_str(val):
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

# ── File reading (raw, no processing) ────────────────────────────────────────
def read_raw(uploaded_file):
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".csv"):
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw_bytes), header=None, dtype=str,
                    keep_default_na=False, encoding=enc,
                    on_bad_lines="skip", engine="python",
                )
                return df
            except Exception:
                continue
    else:
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes), header=None, dtype=str, engine="openpyxl")
        except Exception:
            df = pd.read_excel(io.BytesIO(raw_bytes), header=None, dtype=str, engine="xlrd")
        return df.fillna("")
    return pd.DataFrame()

# ── Apply header + skip settings ─────────────────────────────────────────────
def apply_header(df_raw, skip_rows, header_row):
    """
    skip_rows: number of rows to drop from top (0 = keep all)
    header_row: which row (after skipping) to use as headers (1-indexed)
    """
    df = df_raw.copy()
    # Drop junk rows from top
    df = df.iloc[skip_rows:].reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame(), []

    # header_row is 1-indexed; convert to 0-indexed
    h_idx = header_row - 1
    if h_idx >= len(df):
        return pd.DataFrame(), []

    raw_headers = [str(v).strip() for v in df.iloc[h_idx]]
    headers = [normalize_col(v) for v in raw_headers]

    data = df.iloc[h_idx + 1:].reset_index(drop=True)
    data.columns = headers

    # Drop blank rows
    data = data[data.apply(lambda r: any(str(v).strip() != "" for v in r), axis=1)]

    # Phone → integer string
    if "Phone" in data.columns:
        data["Phone"] = data["Phone"].apply(phone_to_int_str)

    return data, headers

# ── Processing logic ──────────────────────────────────────────────────────────
def process_data(df, markets, lowercase_emails=True):
    if df.empty:
        return pd.DataFrame(), {}, {}, {}

    df = df.copy()
    if lowercase_emails and "Email" in df.columns:
        df["Email"] = df["Email"].str.lower()

    internal, external = [], []
    sorted_codes = sorted(COUNTRY_INFO.items(), key=lambda x: -len(x[0]))

    for _, row in df.iterrows():
        email = str(row.get("Email", "")).strip()
        if not is_valid_email(email):
            continue
        if is_internal(email):
            part = email.split("@")[0].strip()
            fmt = part
            if part.startswith("+"):
                for code, info in sorted_codes:
                    if part[1:].startswith(code):
                        num = digits_only(part[1 + len(code):])
                        if num:
                            fmt = f"{code} {num[-info['length']:]}"
                        break
            internal.append({"Email": email, "Phone": fmt})
        else:
            raw_phone = digits_only(row.get("Phone", ""))
            if len(raw_phone) < 5:
                continue
            currency = str(row.get("currency", "")).strip()
            if markets and currency not in markets:
                continue
            mapping = CURRENCY_MAP.get(currency)
            if not mapping:
                continue
            phone = raw_phone
            if phone.startswith(mapping["code"]):
                phone = phone[len(mapping["code"]):]
            external.append({"Email": email, "Phone": f"{mapping['code']} {phone[:mapping['length']]}"})

    # Deduplicate
    seen, deduped = set(), []
    for r in internal + external:
        key = f"{r['Email']}|{r['Phone']}"
        if key not in seen and r.get("Phone", "").strip():
            seen.add(key)
            deduped.append(r)

    # Group by email
    grouped = defaultdict(list)
    for r in deduped:
        grouped[r["Email"]].append(r["Phone"])

    final = pd.DataFrame(
        [{"Email": e, "Phone": ", ".join(p)} for e, p in grouped.items()]
    ) if grouped else pd.DataFrame(columns=["Email", "Phone"])

    total = len(df)
    ve = df["Email"].apply(lambda e: is_valid_email(str(e))).sum() if "Email" in df.columns else 0
    vp = df["Phone"].apply(lambda p: len(digits_only(p)) >= 5).sum() if "Phone" in df.columns else 0
    ic = df["Email"].apply(lambda e: is_internal(str(e))).sum() if "Email" in df.columns else 0

    stats = {
        "Total Input Rows": total,
        "Valid Emails": int(ve),
        "Valid Phones": int(vp),
        "Internal Emails": int(ic),
        "Output Records": len(final),
    }

    country_split = defaultdict(int)
    for r in deduped:
        code = r["Phone"].split(" ")[0]
        country_split[COUNTRY_INFO.get(code, {}).get("name", "Unknown")] += 1

    currency_split = defaultdict(int)
    if "currency" in df.columns:
        for v in df["currency"]:
            currency_split[str(v).strip() or "Unknown"] += 1

    return final, stats, dict(country_split), dict(currency_split)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("## 📊 DataSync Pro")
st.markdown("PI Data Processing for Targeting")
st.divider()

# ─── STEP 1: Upload ────────────────────────────────────────────────────────
st.markdown('<p class="section-title">① Upload File</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop your CSV or Excel file here",
    type=["csv", "xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded_file:
    df_raw = read_raw(uploaded_file)

    if df_raw.empty:
        st.error("Could not read the file. Try saving it as UTF-8 CSV.")
        st.stop()

    st.divider()

    # ─── STEP 2: Configure rows ──────────────────────────────────────────
    st.markdown('<p class="section-title">② Configure Rows</p>', unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 2])
    with col_cfg1:
        skip_rows = st.number_input(
            "Junk rows to skip from top",
            min_value=0, max_value=max(0, len(df_raw) - 2),
            value=min(1, max(0, len(df_raw) - 2)),
            step=1,
            help="How many rows at the top to throw away before the header"
        )
    with col_cfg2:
        header_row = st.number_input(
            "Header row (after skipping)",
            min_value=1, max_value=max(1, len(df_raw) - int(skip_rows)),
            value=1,
            step=1,
            help="Which row (counting from 1 after skipping) is the header"
        )

    df, headers = apply_header(df_raw, int(skip_rows), int(header_row))

    # Raw preview (collapsible)
    with st.expander("🔍 Raw file preview (first 10 rows)", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

    if df.empty:
        st.warning("No data rows found with current settings. Adjust the row configuration above.")
        st.stop()

    # Parsed preview
    st.markdown(f"**Input Preview** — {len(df):,} rows · Columns: `{'`, `'.join(headers)}`")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.divider()

    # ─── STEP 3: Options + Process ──────────────────────────────────────
    st.markdown('<p class="section-title">③ Processing Options</p>', unsafe_allow_html=True)

    col_opt1, col_opt2 = st.columns([3, 1])
    with col_opt1:
        selected_markets = st.multiselect(
            "Select markets to include",
            list(CURRENCY_MAP.keys()),
            default=["INR", "BDT", "USD"],
        )
    with col_opt2:
        lowercase_emails = st.checkbox("Lowercase emails", value=True)

    process_btn = st.button("🚀 Process Data", type="primary", use_container_width=True)

    if process_btn:
        with st.spinner("Processing..."):
            final_df, stats, country_split, currency_split = process_data(
                df, set(selected_markets), lowercase_emails
            )
        st.session_state["result"] = final_df
        st.session_state["stats"] = stats
        st.session_state["country_split"] = country_split
        st.session_state["currency_split"] = currency_split

    # ─── STEP 4: Results ─────────────────────────────────────────────────
    if "result" in st.session_state and not st.session_state["result"].empty:
        final_df      = st.session_state["result"]
        stats         = st.session_state["stats"]
        country_split = st.session_state["country_split"]
        currency_split= st.session_state["currency_split"]

        st.divider()
        st.markdown('<p class="section-title">④ Results</p>', unsafe_allow_html=True)

        # Stats row
        cols = st.columns(5)
        colors = ["#6366f1", "#22c55e", "#3b82f6", "#f59e0b", "#ec4899"]
        for i, (label, value) in enumerate(stats.items()):
            with cols[i]:
                st.metric(label, f"{value:,}" if isinstance(value, int) else value)

        st.markdown("")

        # Charts
        if country_split or currency_split:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if country_split:
                    cdf = pd.DataFrame(list(country_split.items()), columns=["Country", "Count"]).sort_values("Count", ascending=False)
                    st.markdown("**Country split**")
                    st.bar_chart(cdf.set_index("Country"), height=220)
            with col_c2:
                if currency_split:
                    cudf = pd.DataFrame(list(currency_split.items()), columns=["Currency", "Count"]).sort_values("Count", ascending=False)
                    st.markdown("**Currency split**")
                    st.bar_chart(cudf.set_index("Currency"), height=220)

        st.markdown("")

        # Output preview
        st.markdown(f"**Output — {len(final_df):,} records**")
        st.dataframe(final_df, use_container_width=True, hide_index=True, height=350)

        # Export
        st.markdown("")
        col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 3])
        with col_dl1:
            st.download_button(
                "⬇️ Download CSV",
                final_df.to_csv(index=False).encode("utf-8"),
                "processed_data.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_dl2:
            buf = io.BytesIO()
            final_df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                "⬇️ Download XLSX",
                buf.getvalue(),
                "processed_data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
