"""
Preprocessing pipeline for Greater Manchester property listings (mini dataset)
Outputs:
  1) structured_properties.parquet (or CSV fallback) – cleaned, typed, enriched table
  2) embedding_docs.jsonl – embedding-ready documents (listing card + sentence-window chunks)

This script is self-contained and independent of your RAG code.
"""

import os
import re
import json
import math
import hashlib
import datetime as dt
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------
INPUT_CSV = "manchester_properties_for_sale_mini.csv"   # change path as needed
OUT_STRUCTURED_PARQUET = "structured_properties.parquet"
OUT_STRUCTURED_CSV = "structured_properties.csv"        # fallback if parquet engine unavailable
OUT_DOCS = "embedding_docs.jsonl"

SENT_WIN = 5                 # sentence window size
MAX_CHARS_PER_CHUNK = 900    # soft cap per chunk

# ----------------------------
# Helpers
# ----------------------------
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def to_int(val):
    if pd.isna(val): return None
    if isinstance(val, (int, np.integer)): return int(val)
    s = re.sub(r"[£$,]", "", str(val))
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m: return None
    try: return int(float(m.group(0)))
    except: return None

def to_bool(val):
    if isinstance(val, bool): return val
    if pd.isna(val): return None
    s = str(val).strip().lower()
    if s in {"true","1","yes","y","t"}: return True
    if s in {"false","0","no","n","f"}: return False
    return None

def to_date(val):
    if pd.isna(val): return None
    if isinstance(val, (dt.date, dt.datetime, np.datetime64, pd.Timestamp)):
        return pd.to_datetime(val).date().isoformat()
    s = re.sub(r"(?i)added on\s*", "", str(val).strip())
    try:
        parsed_date = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(parsed_date) or parsed_date is pd.NaT:
            return None
        return parsed_date.date().isoformat()
    except:
        try:
            parsed_date = pd.to_datetime(s, errors="coerce")
            if pd.isna(parsed_date) or parsed_date is pd.NaT:
                return None
            return parsed_date.date().isoformat()
        except:
            return None

def normalize_text(val):
    if pd.isna(val): return None
    return re.sub(r"\s+", " ", str(val).strip())

def geohash_encode(lat: float, lon: float, precision: int = 6) -> str | None:
    if lat is None or lon is None:
        return None
    if isinstance(lat, float) and math.isnan(lat): return None
    if isinstance(lon, float) and math.isnan(lon): return None
    lat_interval = [-90.0, 90.0]; lon_interval = [-180.0, 180.0]
    bits = [16,8,4,2,1]; bit=0; ch=0; geohash=[]; even=True
    while len(geohash) < precision:
        if even:
            mid = sum(lon_interval)/2
            if lon > mid: ch |= bits[bit]; lon_interval[0]=mid
            else: lon_interval[1]=mid
        else:
            mid = sum(lat_interval)/2
            if lat > mid: ch |= bits[bit]; lat_interval[0]=mid
            else: lat_interval[1]=mid
        even = not even
        if bit < 4: bit += 1
        else: geohash.append(_BASE32[ch]); bit=0; ch=0
    return "".join(geohash)

def canonical_id(row: pd.Series) -> str:
    """Stable ID using normalized address + price + bedrooms + branchName."""
    addr = str(row.get("displayAddress") or "").lower().strip()
    price = str(row.get("price") or "")
    beds = str(row.get("bedrooms") or "")
    branch = str(row.get("branchName") or "").lower().strip()
    key = f"{addr}|{price}|{beds}|{branch}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

AMENITY_PATTERNS = {
    "has_parking": r"\b(parking|off[-\s]?road parking|driveway|allocated parking|car\s?space)\b",
    "has_balcony": r"\b(balcony|terrace)\b",
    "has_garden": r"\b(garden|yard)\b",
    "new_build": r"\b(new build|brand new|recently built|newly built)\b",
    "student_friendly": r"\b(student|uni|campus)\b",
    "investment": r"\b(investment|yield|tenanted|tenant in place)\b",
}

def extract_amenities(text: str) -> Dict[str, bool]:
    base = (text or "").lower()
    return {k: bool(re.search(p, base)) for k,p in AMENITY_PATTERNS.items()}

def split_sentences(text: str) -> List[str]:
    if not text: return []
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def sentence_window_chunks(text: str, window=5, max_chars=900) -> List[str]:
    sents = split_sentences(text)
    if not sents: return []
    chunks = []
    for i in range(len(sents)):
        start = max(0, i-window); end = min(len(sents), i+window+1)
        chunk = " ".join(sents[start:end])
        chunks.append(chunk[:max_chars] if len(chunk)>max_chars else chunk)
    # dedupe
    seen=set(); unique=[]
    for c in chunks:
        if c not in seen: seen.add(c); unique.append(c)
    return unique

def recency_band(d: str) -> str:
    if not d: return "unknown"
    try:
        date = dt.date.fromisoformat(d)
        delta = (dt.date.today() - date).days
        if delta < 7: return "<7d"
        if delta < 30: return "7–30d"
        if delta < 90: return "30–90d"
        return "90+d"
    except: return "unknown"

def price_band(price: int) -> str:
    if price is None: return "unknown"
    if price < 50000: return "<50k"
    if price < 100000: return "50–100k"
    if price < 150000: return "100–150k"
    if price < 200000: return "150–200k"
    if price < 300000: return "200–300k"
    return "300k+"

def render_listing_card_text(r: pd.Series) -> str:
    # Compact, human-readable "listing card" text used for embeddings
    ptype = r.get("propertyTypeFullDescription") or r.get("propertySubType") or "Property"
    bedrooms = r.get("bedrooms")
    address = r.get("displayAddress") or "Address unknown"
    price = r.get("price")
    ttype = r.get("transactionType") or r.get("channel") or "buy"
    auction_flag = " (for sale by auction)" if r.get("auction") else ""
    branch = r.get("formattedBranchName") or r.get("branchName") or ""
    branch_loc = r.get("branchLocation") or ""
    first_visible = r.get("firstVisibleDate") or ""
    summary = r.get("summary") or ""
    # features from booleans
    feats = []
    if r.get("has_parking"): feats.append("parking")
    if r.get("has_balcony"): feats.append("balcony")
    if r.get("has_garden"): feats.append("garden")
    if r.get("has_floorplan"): feats.append("floorplan")
    if r.get("has_virtual_tour"): feats.append("virtual tour")
    features_text = ", ".join(feats) if feats else "no notable amenities"
    nimg = r.get("numberOfImages") or 0
    nfp = r.get("numberOfFloorplans") or 0
    nvt = r.get("numberOfVirtualTours") or 0
    text = (f"{ptype}; {bedrooms} bedrooms; {address}. "
            f"Price £{price}; {ttype}{auction_flag}. "
            f"Key features: {features_text}. Summary: {summary}. "
            f"Agent: {branch} ({branch_loc}). First visible: {first_visible}. "
            f"Media: {nimg} images, {nfp} floorplans, {nvt} virtual tours.")
    return re.sub(r"\s+", " ", text).strip()

def build_metadata(r: pd.Series) -> Dict[str, Any]:
    meta = {
        "id": r.get("id"),
        "canonical_id": r.get("canonical_id"),
        "price": r.get("price"),
        "bedrooms": r.get("bedrooms"),
        "auction": bool(r.get("auction")) if not pd.isna(r.get("auction")) else False,
        "has_floorplan": bool(r.get("has_floorplan")) if not pd.isna(r.get("has_floorplan")) else False,
        "has_virtual_tour": bool(r.get("has_virtual_tour")) if not pd.isna(r.get("has_virtual_tour")) else False,
        "has_parking": bool(r.get("has_parking")) if not pd.isna(r.get("has_parking")) else False,
        "property_type": r.get("propertyTypeFullDescription") or r.get("propertySubType"),
        "displayAddress": r.get("displayAddress"),
        "city": r.get("city"),
        "postcode": r.get("postcode"),
        "local_authority": r.get("local_authority"),  # placeholder for future enrichment
        "geohash6": r.get("geohash6"),
        "listingUpdateDate": r.get("listingUpdateDate"),
        "firstVisibleDate": r.get("firstVisibleDate"),
        "channel": r.get("channel"),
        "branchName": r.get("branchName"),
        "branchLocation": r.get("branchLocation"),
        "region_id": r.get("region_id"),
        "price_band": r.get("price_band"),
        "updated_recency_band": r.get("updated_recency_band"),
        "is_active": r.get("is_active", True),
        "source_provenance": r.get("source_provenance"),
    }
    return {k:v for k,v in meta.items() if v is not None and v == v}

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(INPUT_CSV)
df["_row_ix"] = np.arange(len(df))
df["source_provenance"] = os.path.basename(INPUT_CSV)

# ----------------------------
# Schema normalization & cleaning
# ----------------------------
def safe_get(col): 
    return df[col] if col in df.columns else pd.Series([None]*len(df))

df["id"] = safe_get("id").apply(to_int)
df["bedrooms"] = safe_get("bedrooms").apply(to_int)
df["numberOfImages"] = safe_get("numberOfImages").apply(to_int)
df["numberOfFloorplans"] = safe_get("numberOfFloorplans").apply(to_int)
df["numberOfVirtualTours"] = safe_get("numberOfVirtualTours").apply(to_int)

df["price"] = safe_get("price").apply(to_int)
df["displayPrice"] = safe_get("displayPrice").astype(str).replace({"nan": None})

df["auction"] = safe_get("auction").apply(to_bool)
df["onlineViewingsAvailable"] = safe_get("onlineViewingsAvailable").apply(to_bool)
df["premiumListing"] = safe_get("premiumListing").apply(to_bool)
df["featuredProperty"] = safe_get("featuredProperty").apply(to_bool)

df["firstVisibleDate"] = safe_get("firstVisibleDate").apply(to_date)
df["listingUpdateDate"] = safe_get("listingUpdateDate").apply(to_date)

for col in ["summary", "displayAddress", "propertySubType", "propertyTypeFullDescription",
            "formattedBranchName", "branchName", "branchLocation", "transactionType", "channel"]:
    if col in df.columns:
        df[col] = df[col].astype(str).replace({"nan": None}).apply(normalize_text)

if "latitude" in df.columns:
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
if "longitude" in df.columns:
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# UK bounds sanity check
if "latitude" in df.columns:
    df.loc[~df["latitude"].between(49, 61), "latitude"] = np.nan
if "longitude" in df.columns:
    df.loc[~df["longitude"].between(-8.6, 2), "longitude"] = np.nan

# ----------------------------
# Derived flags & amenities
# ----------------------------
df["has_floorplan"] = (df["numberOfFloorplans"].fillna(0) > 0)
df["has_virtual_tour"] = (df["numberOfVirtualTours"].fillna(0) > 0)

amenity_text = (df["summary"].fillna("") + " " + df.get("propertyTypeFullDescription", pd.Series([""]*len(df))).fillna(""))
amenities = amenity_text.apply(extract_amenities).apply(pd.Series)
for col in amenities.columns:
    df[col] = amenities[col]

# basic city/postcode heuristics
def extract_city(addr: str) -> str | None:
    if not addr:
        return None
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    if parts:
        for token in reversed(parts):
            if not re.match(r"[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2}$", token, re.I):
                return token.title()
    return None

def extract_postcode(addr: str) -> str | None:
    if not addr:
        return None
    m = re.search(r"\b[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2}\b", addr, re.I)
    return m.group(0).upper().replace(" ", "") if m else None

df["city"] = df["displayAddress"].apply(extract_city)
df["postcode"] = df["displayAddress"].apply(extract_postcode)

# geohash (if lat/lon available)
if "latitude" in df.columns and "longitude" in df.columns:
    def encode_geohash_row(row):
        return geohash_encode(row.get("latitude"), row.get("longitude"), precision=6)
    df["geohash6"] = df.apply(encode_geohash_row, axis=1)
else:
    df["geohash6"] = None

# derived metrics
df["price_band"] = df["price"].apply(price_band)
df["updated_recency_band"] = df["listingUpdateDate"].apply(recency_band)
df["has_media_score"] = (df["has_floorplan"].astype(int) + df["has_virtual_tour"].astype(int) + np.log1p(df["numberOfImages"].fillna(0)))

# active flag
df["is_active"] = True

# canonical ID for dedup
df["canonical_id"] = df.apply(canonical_id, axis=1)

# ----------------------------
# Deduplication: keep most recently updated per canonical_id
# ----------------------------
def iso_to_date_safe(x):
    try: return dt.date.fromisoformat(x) if x else dt.date.min
    except: return dt.date.min

df["__upd_date"] = df["listingUpdateDate"].apply(iso_to_date_safe)
df = df.sort_values(["canonical_id", "__upd_date"], ascending=[True, False]) \
       .drop_duplicates(subset=["canonical_id"], keep="first") \
       .drop(columns=["__upd_date"])

# ----------------------------
# Structured View
# ----------------------------
structured_cols = [
    "id","canonical_id","is_active","source_provenance","_row_ix",
    "displayAddress","city","postcode","latitude","longitude","geohash6",
    "propertySubType","propertyTypeFullDescription","bedrooms","summary",
    "price","displayPrice","transactionType","channel","auction",
    "numberOfImages","numberOfFloorplans","numberOfVirtualTours",
    "has_floorplan","has_virtual_tour",
    "has_parking","has_balcony","has_garden","new_build","student_friendly","investment",
    "formattedBranchName","branchName","branchLocation",
    "firstVisibleDate","listingUpdateDate","updated_recency_band",
    "price_band","has_media_score","region_id"
]
structured_cols = [c for c in structured_cols if c in df.columns]
structured_df = df[structured_cols].copy()

# Try parquet, fallback to CSV if engine missing
try:
    structured_df.to_parquet(OUT_STRUCTURED_PARQUET, index=False)
    print(f"Saved structured table → {OUT_STRUCTURED_PARQUET}")
except Exception as e:
    print("Parquet engine not available, saving CSV instead:", e)
    structured_df.to_csv(OUT_STRUCTURED_CSV, index=False)
    print(f"Saved structured table → {OUT_STRUCTURED_CSV}")

# ----------------------------
# Embedding View (docs.jsonl)
# ----------------------------
with open(OUT_DOCS, "w", encoding="utf-8") as f:
    count = 0
    for _, r in df.iterrows():
        card_text = render_listing_card_text(r)
        chunks = sentence_window_chunks(card_text, window=SENT_WIN, max_chars=MAX_CHARS_PER_CHUNK)
        meta = build_metadata(r)
        # Whole-card chunk
        whole = card_text[:MAX_CHARS_PER_CHUNK]
        f.write(json.dumps({"id": r["id"], "text": whole, "metadata": meta}, ensure_ascii=False) + "\n")
        count += 1
        # Sentence-window chunks (dedup vs whole)
        for c in chunks:
            if c != whole:
                f.write(json.dumps({"id": r["id"], "text": c, "metadata": meta}, ensure_ascii=False) + "\n")
                count += 1

print(f"Saved embedding docs → {OUT_DOCS}")
print("Done.")
