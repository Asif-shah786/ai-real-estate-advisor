"""
Preprocessing pipeline for Dataset 2 (Zoopla-style JSON-flattened CSV)
Author: Data Science (RAG) Team
Purpose:
  - Clean, normalise, and enrich rich listing records for a RAG application.
  - Produce TWO artifacts:
      (1) structured_properties_v2.parquet/.csv  -> exact, typed fields for filters & factual lookups
      (2) embedding_docs_v2.jsonl                -> compact "listing card" texts + sentence-window chunks for embeddings
Usage:
  python preprocess_v2.py --input /path/to/16-august-4pm_page1-ALL_property1-19.csv --outdir ./artifacts
Notes:
  - Self-contained; no Streamlit/LangChain dependency.
  - Safe on mixed/dirty fields. Handles booleans, numerics, EPC, council tax, tenure, room sizes, URLs.
"""

from __future__ import annotations
import os, re, json, math, argparse, hashlib
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

# ----------------------------
# Helpers
# ----------------------------
GBP_RE = re.compile(r"[£,\s]")
DIM_RE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*(m|metre|meter|ft|foot|feet)\b")
ROOM_DIM_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*m\s*x\s*(\d+(?:\.\d+)?)\s*m")
POSTCODE_RE = re.compile(r"(?i)\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s?(\d[A-Z]{2})\b")
OUTCODE_RE = re.compile(r"(?i)\b([A-Z]{1,2}\d{1,2}[A-Z]?)\b")

def to_bool(x) -> Optional[bool]:
    if isinstance(x, bool): return x
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
    s = str(x).strip().lower()
    if s in {"true","t","1","yes","y"}: return True
    if s in {"false","f","0","no","n"}: return False
    return None

def to_int(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
    s = GBP_RE.sub("", str(x))
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m: return None
    try:
        val = float(m.group(0))
        return int(round(val))
    except:
        return None

def to_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
    s = GBP_RE.sub("", str(x))
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m: return None
    try:
        return float(m.group(0))
    except:
        return None

def norm_text(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
    return re.sub(r"\s+", " ", str(x).strip())

def parse_date(x) -> Optional[str]:
    if x is None: return None
    try:
        d = pd.to_datetime(x, dayfirst=True, errors="coerce")
        if pd.isna(d): return None
        return d.date().isoformat()
    except Exception:
        return None

def extract_postcode_address(addr: str) -> Tuple[Optional[str], Optional[str]]:
    if not addr: return None, None
    m_pc = POSTCODE_RE.search(addr.upper())
    if m_pc:
        postcode = f"{m_pc.group(1)} {m_pc.group(2)}"
        outcode = m_pc.group(1)
        return postcode, outcode
    # fallback using outcode if full postcode missing
    m_out = OUTCODE_RE.search(addr.upper())
    return None, (m_out.group(1) if m_out else None)

def canonical_id(row: pd.Series) -> str:
    addr = (row.get("display_address") or row.get("address") or "").lower()
    price = row.get("price") or ""
    agent = (row.get("agent") or "").lower()
    key = f"{addr}|{price}|{agent}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def epc_to_band(x: str) -> Optional[str]:
    if not x: return None
    s = str(x).strip().upper()
    m = re.match(r"^[A-G]", s)
    return m.group(0) if m else None

def council_band(x: str) -> Optional[str]:
    if not x: return None
    s = str(x).strip().upper()
    m = re.match(r"^[A-H]", s)
    return m.group(0) if m else None

def tenure_norm(x: str) -> Optional[str]:
    if not x: return None
    s = str(x).strip().lower()
    if "freehold" in s: return "freehold"
    if "leasehold" in s: return "leasehold"
    if "commonhold" in s: return "commonhold"
    return s or None

AMENITY_PATTERNS = {
    "has_parking": r"\b(parking|off[-\s]?road parking|driveway|allocated parking|garage|carport|car\s?space)\b",
    "has_garden": r"\b(garden|yard|courtyard)\b",
    "near_park": r"\b(park|greenspace|green space)\b",
    "near_transport": r"\b(tram|metrolink|station|rail|train|bus stop)\b",
}

def extract_amenities(text: str) -> Dict[str, bool]:
    base = (text or "").lower()
    return {k: bool(re.search(p, base)) for k, p in AMENITY_PATTERNS.items()}

def parse_room_dim_m(s: str) -> Optional[Tuple[float,float]]:
    if not s or not isinstance(s, str): return None
    m = ROOM_DIM_RE.search(s.replace("×", "x"))
    if not m: return None
    try:
        return float(m.group(1)), float(m.group(2))
    except: 
        return None

def metres_from_dim_str(s: str) -> Optional[float]:
    """Return area (m^2) if string looks like '4.17m x 3.20m' else None."""
    dims = parse_room_dim_m(s)
    if not dims: return None
    return round(dims[0]*dims[1], 2)

def sentence_split(text: str) -> List[str]:
    if not text: return []
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def sentence_window_chunks(text: str, window=5, max_chars=900) -> List[str]:
    sents = sentence_split(text)
    if not sents: return []
    chunks = []
    for i in range(len(sents)):
        lo = max(0, i-window); hi = min(len(sents), i+window+1)
        chunk = " ".join(sents[lo:hi])
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars]
        chunks.append(chunk)
    # dedupe while keeping order
    seen=set(); out=[]
    for c in chunks:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def compute_source(row: pd.Series) -> str:
    """Compute a safe source string for metadata"""
    url = row.get("property_url")
    if url and isinstance(url, str) and url.strip():
        return url.strip()
    prov = (row.get("source_provenance") or row.get("source") or "unknown").strip()
    ds = (row.get("__dataset_file") or "dataset.csv").strip()
    lid = str(row.get("listing_id") or row.get("canonical_id") or "unknown").strip()
    return f"{prov}|{ds}#{lid}"

def build_listing_card(row: pd.Series) -> str:
    ptype = row.get("property_type") or "Property"
    beds = row.get("bedrooms")
    baths = row.get("bathrooms")
    addr = row.get("display_address") or row.get("address") or "Address unknown"
    price = row.get("price")
    tenure = row.get("tenure")
    status = row.get("status")
    agent = row.get("agent")
    epc = row.get("epc_band")
    council = row.get("council_tax_band")
    size = row.get("size_sq_feet")
    nphotos = row.get("number_of_photos") or 0
    nfp = row.get("number_of_floorplans") or 0
    has_fp = row.get("has_floorplan")
    chain_free = row.get("chain_free")
    feats = []
    if row.get("has_parking"): feats.append("parking")
    if row.get("has_garden"): feats.append("garden")
    if row.get("has_epc"): feats.append("EPC available")
    if has_fp: feats.append("floorplan")
    feat_text = ", ".join(feats) if feats else "no notable amenities"
    summary = row.get("about_property") or ""
    text = (
        f"{ptype}; {beds} bedrooms, {baths} bathrooms; {addr}. "
        f"Price £{price}; tenure {tenure}; status {status}; agent {agent}. "
        f"EPC {epc}; council tax band {council}; size {size} sq ft. "
        f"Key features: {feat_text}. Summary: {summary}"
    )
    return re.sub(r"\s+", " ", text).strip()

def build_metadata(row: pd.Series) -> Dict[str, Any]:
    meta = {
        "id": row.get("listing_id"),
        "canonical_id": row.get("canonical_id"),
        "price": row.get("price"),
        "bedrooms": row.get("bedrooms"),
        "bathrooms": row.get("bathrooms"),
        "receptions": row.get("receptions"),
        "tenure": row.get("tenure"),
        "chain_free": bool(row.get("chain_free")) if row.get("chain_free") is not None else False,
        "has_floorplan": bool(row.get("has_floorplan")) if row.get("has_floorplan") is not None else False,
        "has_epc": bool(row.get("has_epc")) if row.get("has_epc") is not None else False,
        "epc_band": row.get("epc_band"),
        "council_tax_band": row.get("council_tax_band"),
        "property_type": row.get("property_type"),
        "display_address": row.get("display_address") or row.get("address"),
        "outcode": row.get("outcode"),
        "postcode": row.get("postcode"),
        "price_per_sqft": row.get("price_per_sqft"),
        "size_sq_feet": row.get("size_sq_feet"),
        "source": compute_source(row),  # Use computed safe source
        "source_title": (row.get("title") or f"Listing {row.get('listing_id') or row.get('canonical_id') or ''}").strip(),
        "property_url": row.get("property_url"),
        "agent": row.get("agent"),
        "status": row.get("status"),
        "scraped_at": row.get("scraped_at"),
    }
    return {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k,v in meta.items()}

# ----------------------------
# Main
# ----------------------------
def run(input_csv: str, outdir: str):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    
    # Capture dataset file name for source tracking
    src_file = Path(input_csv).name
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    df["__dataset_file"] = src_file

    # ---- Normalise core types
    if "listing_id" not in df.columns and "property_id" in df.columns:
        df.rename(columns={"property_id":"listing_id"}, inplace=True)
    elif "listing_id" not in df.columns:
        df["listing_id"] = df.get("property_id")

    int_cols = ["price","bedrooms","bathrooms","receptions","number_of_photos","number_of_floorplans","size_sq_feet","price_per_sqft"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].apply(to_int)

    bool_cols = ["has_epc","has_floorplan","chain_free"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].apply(to_bool)

    text_cols = ["display_address","address","title","status","tenure","property_type","agent","epc_rating","council_tax_band","outcode","about_property","property_url","source","nearest_stations","nearest_schools"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].apply(norm_text)

    # ---- Derived fields
    # EPC band letter
    if "epc_rating" in df.columns:
        df["epc_band"] = df["epc_rating"].apply(epc_to_band)
    elif "epc_band" not in df.columns:
        df["epc_band"] = None

    # council tax band
    if "council_tax_band" in df.columns:
        df["council_tax_band"] = df["council_tax_band"].apply(council_band)

    # tenure
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].apply(tenure_norm)

    # outcode/postcode from address
    if "postcode" not in df.columns or df["postcode"].isna().all():
        # try extract from address/display_address
        pcs = []; ocs=[]
        for a1, a2 in zip(df.get("display_address", [None]*len(df)),
                          df.get("address", [None]*len(df))):
            pc, oc = extract_postcode_address(a1 or a2 or "")
            pcs.append(pc); ocs.append(oc)
        df["postcode"] = pcs
        df["outcode"] = df.get("outcode") if "outcode" in df.columns else ocs
        df["outcode"] = df["outcode"].fillna(pd.Series(ocs, index=df.index))

    # compute price per sqft if possible
    if "price_per_sqft" not in df.columns and all(c in df.columns for c in ["price","size_sq_feet"]):
        df["price_per_sqft"] = np.where(df["price"].notna() & df["size_sq_feet"].notna() & (df["size_sq_feet"]>0),
                                        (df["price"] / df["size_sq_feet"]).round(0),
                                        np.nan)

    # amenities from about_property
    amen_df = df.get("about_property", pd.Series([None]*len(df))).apply(lambda s: extract_amenities(s or ""))
    amen_df = pd.DataFrame(list(amen_df.values), index=df.index) if not isinstance(amen_df, pd.DataFrame) else amen_df
    for col in ["has_parking","has_garden","near_park","near_transport"]:
        if col not in amen_df.columns:
            amen_df[col] = False
    df = pd.concat([df, amen_df[["has_parking","has_garden","near_park","near_transport"]]], axis=1)

    # canonical id
    df["canonical_id"] = df.apply(canonical_id, axis=1)

    # recentness band from 'scraped_at' or status
    def recency_band(x):
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return "unknown"
        # Handle timezone-aware datetime objects
        if d.tz is not None:
            d = d.tz_localize(None)
        delta = (pd.Timestamp.now() - d).days
        if delta <= 7: return "added_7d"
        if delta <= 30: return "added_30d"
        return "older"
    df["updated_recency_band"] = df.get("scraped_at", pd.Series([None]*len(df))).apply(recency_band)

    # source provenance
    df["source_provenance"] = df.get("source", pd.Series(["unknown"]*len(df)))

    # ---- Structured view columns
    structured_cols = [
        "listing_id","canonical_id","display_address","postcode","outcode",
        "price","price_per_sqft","bedrooms","bathrooms","receptions",
        "tenure","chain_free","property_type","status",
        "has_floorplan","number_of_floorplans","has_epc","epc_band",
        "council_tax_band","size_sq_feet","number_of_photos",
        "has_parking","has_garden","near_park","near_transport",
        "agent","property_url","source_provenance","scraped_at","updated_recency_band","about_property","title"
    ]
    structured_cols = [c for c in structured_cols if c in df.columns]
    structured = df[structured_cols].copy()

    # save structured
    out_parquet = out / "structured_properties_v2.parquet"
    out_csv = out / "structured_properties_v2.csv"
    try:
        structured.to_parquet(out_parquet, index=False)
        print(f"Saved structured → {out_parquet}")
    except Exception as e:
        structured.to_csv(out_csv, index=False)
        print(f"Parquet engine missing; saved CSV → {out_csv} ({e})")

    # ---- Build embedding docs
    docs_path = out / "embedding_docs_v2.jsonl"
    count = 0
    with docs_path.open("w", encoding="utf-8") as f:
        for _, r in structured.iterrows():
            card = build_listing_card(r)
            meta = build_metadata(r)
            
            # Assert that source is non-empty before writing
            assert isinstance(meta.get("source"), str) and meta["source"].strip(), \
                f"Empty metadata['source'] for listing_id={r.get('listing_id')}"
            
            # Whole-card chunk
            whole = card[:900]
            f.write(json.dumps({"id": r.get("listing_id"), "text": whole, "metadata": meta}, ensure_ascii=False) + "\n")
            count += 1
            # sentence-window chunks from about_property for richer coverage
            for chunk in sentence_window_chunks(r.get("about_property") or "", window=5, max_chars=900):
                if chunk and chunk != whole:
                    f.write(json.dumps({"id": r.get("listing_id"), "text": chunk, "metadata": meta}, ensure_ascii=False) + "\n")
                    count += 1
    print(f"Saved {count} embedding docs → {docs_path}")
    
    # Quick validation: no empty sources
    print("Validating JSONL output...")
    with docs_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            obj = json.loads(line)
            src = obj.get("metadata", {}).get("source", "")
            assert isinstance(src, str) and src.strip(), f"Empty source at line {i}"
    print("✅ Validation OK: all docs include non-empty metadata['source'].")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Dataset 2 CSV")
    ap.add_argument("--outdir", default="artifacts_v2", help="Output directory for artifacts")
    args = ap.parse_args()
    run(args.input, args.outdir)
