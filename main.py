import streamlit as st
import pandas as pd
import re
import os
from fuzzywuzzy import process

st.set_page_config(page_title="🏠 Smart AI Property Chatbot", layout="wide")

# -------------------------------
# Load CSVs
# -------------------------------
@st.cache_data(show_spinner=False)
def load_csvs():
    dfs = []
    for f in os.listdir("."):
        if f.lower().endswith(".csv"):
            try:
                df = pd.read_csv(f, low_memory=False)
                df.columns = df.columns.str.strip().str.lower()
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        return pd.DataFrame()

    base = dfs[0].fillna("")
    for other in dfs[1:]:
        common = set(base.columns) & set(other.columns)
        if "project_id" in common:
            base = base.merge(other, on="project_id", how="left")
        else:
            base = pd.concat([base.reset_index(drop=True), other.reset_index(drop=True)], axis=1)
    base.columns = base.columns.str.lower().str.strip()
    return base.fillna("")

# -------------------------------
# Utilities
# -------------------------------
COMMON_CITIES = [
    "mumbai", "pune", "delhi", "bangalore", "bengaluru",
    "hyderabad", "chennai", "kolkata", "ahmedabad", "noida", "gurgaon", "thane"
]

def fuzzy_city(query):
    match, score = process.extractOne(query.lower(), COMMON_CITIES)
    if score > 70:
        return match
    return None

def parse_price_token(txt):
    txt = txt.lower().replace(",", "").strip()
    m = re.search(r"([\d\.]+)\s*(cr|crore)", txt)
    if m: return float(m.group(1)) * 1e7
    m = re.search(r"([\d\.]+)\s*(l|lac|lakh|l)", txt)
    if m: return float(m.group(1)) * 1e5
    m = re.search(r"([\d\.]+)", txt)
    if m:
        val = float(m.group(1))
        if val < 1000:
            return val * 1e5
        return val
    return None

def parse_query(q):
    q = q.lower().strip()

    # fuzzy detect city
    city = fuzzy_city(q)
    bhk = None
    m = re.search(r"(\d+)\s*bhk", q)
    if m: bhk = int(m.group(1))

    # price understanding
    m = re.search(r"(?:under|below|upto|less than)\s*([\d\.]+\s*(?:cr|crore|lakh|lac|l)?)", q)
    max_price = parse_price_token(m.group(1)) if m else None

    # Detect intent keywords
    intents = {
        "cheap": ["cheap", "budget", "affordable", "low price", "under", "economy"],
        "luxury": ["luxury", "high end", "premium", "expensive"],
        "best": ["best", "top", "recommended", "popular", "famous"],
        "near": ["near", "around", "close to", "beside", "nearby"],
        "family": ["family", "kids", "safe", "peaceful", "residential"],
        "investment": ["investment", "returns", "roi", "profit"],
    }

    detected = []
    for k, words in intents.items():
        for w in words:
            if w in q:
                detected.append(k)
                break

    # handle limit
    m = re.search(r"(top|best)\s*(\d+)", q)
    limit = int(m.group(2)) if m else 5

    return {"city": city, "bhk": bhk, "max_price": max_price, "intents": detected, "limit": limit}

def to_price(x):
    if not isinstance(x, str): return None
    return parse_price_token(x)

def search_properties(df, filters):
    df = df.copy()
    df.columns = df.columns.str.lower()

    price_col = next((c for c in df.columns if "price" in c or "amount" in c), None)
    city_col = next((c for c in df.columns if "city" in c or "location" in c or "area" in c), None)
    name_col = next((c for c in df.columns if "projectname" in c or "name" in c or "title" in c), df.columns[0])
    addr_col = next((c for c in df.columns if "address" in c), None)
    desc_col = next((c for c in df.columns if "description" in c or "about" in c), None)

    if price_col:
        df["_price"] = df[price_col].astype(str).apply(to_price)
    else:
        df["_price"] = None

    mask = pd.Series(True, index=df.index)

    # city
    if filters.get("city"):
        mask &= df.apply(lambda r: any(filters["city"].lower() in str(v).lower() for v in r.astype(str)), axis=1)

    # bhk
    if filters.get("bhk"):
        n = filters["bhk"]
        pattern = re.compile(rf"\b{n}\s*bhk\b", re.IGNORECASE)
        mask &= df.apply(lambda r: any(bool(pattern.search(str(v))) for v in r.astype(str)), axis=1)

    # max price
    if filters.get("max_price"):
        mask &= df["_price"].fillna(1e15) <= filters["max_price"]

    results = df[mask]

    if results.empty:
        return "😕 Sorry, no matching properties found. Try using simpler terms.", None

    # intent-based sorting
    if "cheap" in filters.get("intents", []):
        results = results.sort_values("_price", ascending=True)
    elif "luxury" in filters.get("intents", []):
        results = results.sort_values("_price", ascending=False)
    elif "best" in filters.get("intents", []):
        results = results.sample(frac=1)

    results = results.head(filters["limit"])

    # natural response
    if "cheap" in filters.get("intents", []):
        intro = "💸 Here are some budget-friendly options:"
    elif "luxury" in filters.get("intents", []):
        intro = "💎 These are some luxury apartments you might like:"
    elif "best" in filters.get("intents", []):
        intro = "🏆 Top properties based on popularity and data:"
    else:
        intro = "🏠 Here are some matching properties:"

    lines = []
    for _, r in results.iterrows():
        name = str(r.get(name_col, "")).strip()
        price = str(r.get(price_col, "")).strip() if price_col else ""
        city = str(r.get(city_col, "")).strip() if city_col else ""
        addr = str(r.get(addr_col, "")).strip() if addr_col else ""
        desc = str(r.get(desc_col, "")).strip() if desc_col else ""
        desc = desc[:100] + "..." if len(desc) > 100 else desc

        msg = f"🏢 **{name}**"
        if city: msg += f" — {city.title()}"
        if price: msg += f" (Price: {price})"
        if addr: msg += f"\n📍 {addr}"
        if desc: msg += f"\n💬 {desc}"
        lines.append(msg)

    reply = intro + "\n\n" + "\n\n".join(lines)
    return reply, results

# -------------------------------
# Chat UI
# -------------------------------
data = load_csvs()
st.title("🏠 Ultra Smart Property Chatbot")
st.caption("Ask me anything naturally: e.g. 'cheapest 2bhk in mumbay under 1cr', 'top 3 luxury flats in pune', 'family flat near delhi'")

if data.empty:
    st.error("No CSV file found! Put your CSVs in this folder.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["text"])

if prompt := st.chat_input("Ask about properties..."):
    st.session_state.chat_history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🔍 Understanding your query..."):
        filters = parse_query(prompt)
        reply, _ = search_properties(data, filters)

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "text": reply})
