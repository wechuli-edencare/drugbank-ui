import json
import logging
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import streamlit as st
import pandas as pd

# ----------------------------
# LOGGING SETUP
# ----------------------------
def setup_logging():
    """Setup context-aware logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('drugbank_app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('drugbank_ui')

logger = setup_logging()

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = Path("output.parquet")  # change if your file is elsewhere
APP_TITLE = "DrugBank"

# Comprehensive field mapping based on analysis
CORE_FIELDS = [
    "name", "drugbank-id", "cas-number", "unii", "state", "groups"
]

SUMMARY_FIELDS = [
    "name", "drugbank-id", "cas-number", "unii", "state", "groups",
    "indication", "mechanism-of-action", "description", "pharmacodynamics"
]

DETAILED_FIELDS = [
    "categories", "targets", "enzymes", "carriers", "transporters",
    "pathways", "reactions", "drug-interactions", "food-interactions",
    "absorption", "metabolism", "toxicity", "half-life", "clearance",
    "protein-binding", "volume-of-distribution", "route-of-elimination"
]

SEARCHABLE_FIELDS = [
    "name", "drugbank-id", "cas-number", "unii", "synonyms", 
    "international-brands", "products", "description", "indication",
    "mechanism-of-action", "categories"
]

st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")


# ----------------------------
# UTILITIES
# ----------------------------
JsonVal = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]

def as_list(v: JsonVal) -> List[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]

def textify(v: JsonVal) -> str:
    """Convert JSON value to a human-readable string."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, dict):
        # prefer '#text' if present
        if "#text" in v and isinstance(v["#text"], (str, int, float, bool)):
            return str(v["#text"])
        # otherwise join scalar values
        parts: List[str] = []
        for k, val in v.items():
            if k == "@attributes":
                continue
            s = textify(val)
            if s:
                parts.append(s)
        return " | ".join([p for p in parts if p])
    if isinstance(v, list):
        return "; ".join([textify(x) for x in v if textify(x)])
    return ""

def collect_texts(values: Iterable[JsonVal]) -> List[str]:
    out: List[str] = []
    for v in values:
        s = textify(v).strip()
        if s:
            out.append(s)
    return out

def flatten_ids(drug: Dict[str, Any]) -> List[str]:
    raw = drug.get("drugbank-id")
    ids: List[str] = []
    for v in as_list(raw):
        if isinstance(v, dict):
            if "#text" in v:
                ids.append(str(v["#text"]))
            else:
                ids.append(textify(v))
        else:
            ids.append(textify(v))
    return [x for x in ids if x]

def extract_groups(drug: Dict[str, Any]) -> List[str]:
    """
    DrugBank's <groups><group>approved</group>...</groups> can arrive as:
    - "groups": [{"group": "approved"}, {"group": "withdrawn"}]
    - "groups": [{"group": {"#text": "approved"}}, ...]
    - "groups": {"group": ...}
    We normalize all of these to a simple list of strings.
    """
    raw = drug.get("groups")
    vals: List[str] = []

    if raw is None:
        return vals

    # Common cases
    if isinstance(raw, dict) and "group" in raw:
        entries = as_list(raw["group"])
        vals.extend([textify(x) for x in entries])

    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "group" in item:
                entries = as_list(item["group"])
                vals.extend([textify(x) for x in entries])
            else:
                vals.append(textify(item))
    else:
        vals.append(textify(raw))

    return [v for v in [v.strip() for v in vals] if v]

def extract_synonyms(drug: Dict[str, Any]) -> List[str]:
    # Synonyms may live under multiple places in DrugBank; we grab a few likely ones
    candidates: List[str] = []
    for key in ["synonyms", "international-brands", "brands", "products", "products-list", "names"]:
        v = drug.get(key)
        if v is None:
            continue
        candidates.extend(as_list(v))
    # Also scan a few common substructures: {"synonym": ...}, {"name": ...}, {"brand": ...}
    out: List[str] = []
    for item in candidates:
        if isinstance(item, dict):
            # Look for subkeys that are likely to hold names
            for subk in ["synonym", "name", "brand", "product", "international-brand"]:
                if subk in item:
                    out.extend([textify(x) for x in as_list(item[subk])])
            # As a fallback, textify the dict
            if not any(k in item for k in ["synonym", "name", "brand", "product", "international-brand"]):
                out.append(textify(item))
        else:
            out.append(textify(item))
    # dedupe
    seen = set()
    uniq = []
    for s in [x.strip() for x in out if x and x.strip()]:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq

def safe_get(drug: Dict[str, Any], key: str) -> str:
    return textify(drug.get(key)).strip()

def searchable_text(drug: Dict[str, Any]) -> str:
    """
    Build comprehensive searchable text from all relevant fields.
    """
    parts: List[str] = []
    
    # Core identifiers
    parts.append(safe_get(drug, "name"))
    parts.extend(flatten_ids(drug))
    for k in ["cas-number", "unii"]:
        parts.append(safe_get(drug, k))
    
    # Synonyms and brands
    parts.extend(extract_synonyms(drug))
    
    # Additional searchable content
    for field in ["description", "indication", "mechanism-of-action", "pharmacodynamics"]:
        content = safe_get(drug, field)
        if content:
            # Truncate very long fields for search index
            parts.append(content[:500] if len(content) > 500 else content)
    
    # Categories
    categories = extract_categories(drug)
    parts.extend(categories)
    
    return " | ".join([p for p in parts if p]).lower()

def extract_categories(drug: Dict[str, Any]) -> List[str]:
    """Extract drug categories for search and display"""
    categories = drug.get("categories", [])
    if not categories:
        return []
    
    result = []
    for cat in as_list(categories):
        if isinstance(cat, dict):
            # Look for category name in various possible keys
            for key in ["category", "name", "#text"]:
                if key in cat:
                    result.append(textify(cat[key]))
                    break
            else:
                result.append(textify(cat))
        else:
            result.append(textify(cat))
    
    return [c for c in result if c]

def extract_detailed_field(drug: Dict[str, Any], field: str) -> Dict[str, Any]:
    """Extract and structure detailed field information"""
    data = drug.get(field)
    if not data:
        return {"summary": "Not available", "details": []}
    
    if field == "targets":
        return extract_targets(data)
    elif field == "drug-interactions":
        return extract_drug_interactions(data)
    elif field == "categories":
        return {"summary": f"{len(extract_categories(drug))} categories", "details": extract_categories(drug)}
    else:
        # Generic handler
        text = textify(data)
        return {
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "details": [text] if text else []
        }

def extract_targets(targets_data: Any) -> Dict[str, Any]:
    """Extract target information"""
    if not targets_data:
        return {"summary": "No targets", "details": []}
    
    targets = as_list(targets_data)
    details = []
    
    for target in targets:
        if isinstance(target, dict):
            name = safe_get(target, "name") or "Unknown target"
            actions = target.get("actions", {})
            action_list = []
            if actions:
                action_list = [textify(a) for a in as_list(actions.get("action", []))]
            
            details.append({
                "name": name,
                "actions": action_list,
                "organism": safe_get(target, "organism"),
                "id": safe_get(target, "id")
            })
    
    return {
        "summary": f"{len(details)} target(s)",
        "details": details
    }

def extract_drug_interactions(interactions_data: Any) -> Dict[str, Any]:
    """Extract drug interaction information"""
    if not interactions_data:
        return {"summary": "No known interactions", "details": []}
    
    interactions = as_list(interactions_data)
    details = []
    
    for interaction in interactions:
        if isinstance(interaction, dict):
            drug_name = safe_get(interaction, "name") or safe_get(interaction, "drug")
            description = safe_get(interaction, "description")
            
            details.append({
                "drug": drug_name,
                "description": description
            })
    
    return {
        "summary": f"{len(details)} interaction(s)",
        "details": details
    }


# ----------------------------
# DATA LOADING (cached)
# ----------------------------

DATA_PATH = Path("output.parquet")  # now points to Parquet

@st.cache_data(show_spinner=True)
def load_data(path: Path) -> list[dict]:
    start_time = datetime.now()
    logger.info(f"Loading drug data from {path}")
    try:
        df = pd.read_parquet(path)
        data = df.to_dict(orient="records")
        load_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Successfully loaded {len(data)} drugs in {load_time:.2f}s")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

@st.cache_data
def build_index(rows: List[Dict[str, Any]]) -> List[str]:
    """Build search index with logging"""
    start_time = datetime.now()
    logger.info("Building search index...")
    
    index = [searchable_text(d) for d in rows]
    
    build_time = (datetime.now() - start_time).total_seconds()
    avg_length = sum(len(text) for text in index) / len(index) if index else 0
    logger.info(f"Built search index in {build_time:.2f}s, avg text length: {avg_length:.0f} chars")
    
    return index

@st.cache_data
def get_field_statistics(drugs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze field availability across all drugs"""
    field_counts = {}
    total_drugs = len(drugs)
    
    # Sample a subset for performance
    sample_size = min(1000, total_drugs)
    sample = drugs[:sample_size]
    
    for drug in sample:
        for key in drug.keys():
            field_counts[key] = field_counts.get(key, 0) + 1
    
    # Calculate percentages and sort by frequency
    field_stats = {
        field: {
            "count": count,
            "percentage": (count / sample_size) * 100
        }
        for field, count in field_counts.items()
    }
    
    return {
        "total_drugs": total_drugs,
        "sample_size": sample_size,
        "fields": dict(sorted(field_stats.items(), key=lambda x: x[1]["percentage"], reverse=True))
    }


# ----------------------------
# UI
# ----------------------------
st.title(APP_TITLE)
st.caption("Local, read-only viewer for your converted DrugBank JSON")

if not DATA_PATH.exists():
    st.error(f"Could not find data file at: {DATA_PATH.resolve()}")
    st.stop()

with st.spinner("Loading data..."):
    drugs = load_data(DATA_PATH)

# Build indexes and stats
index_text = build_index(drugs)
field_stats = get_field_statistics(drugs)

# Display data insights
with st.expander("📊 Dataset Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Drugs", field_stats["total_drugs"])
    with col2:
        st.metric("Unique Fields", len(field_stats["fields"]))
    with col3:
        complete_entries = sum(1 for d in drugs[:100] if len(d.keys()) > 20)  # Sample
        st.metric("Rich Entries (sample)", f"{complete_entries}%")
    
    # Field availability
    st.subheader("Field Availability (top 20)")
    top_fields = list(field_stats["fields"].items())[:20]
    for field, stats in top_fields:
        st.write(f"**{field}**: {stats['percentage']:.1f}% ({stats['count']}/{field_stats['sample_size']})")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Search & Filters")
    
    # Search options
    search_mode = st.radio("Search Mode", ["Basic", "Advanced", "Field-specific"])
    
    if search_mode == "Basic":
        q = st.text_input("Search drugs", placeholder="e.g., lepirudin, DB00001, aspirin").strip()
        use_regex = False
    elif search_mode == "Advanced":
        q = st.text_input("Regex Search", placeholder="e.g., (?i)heparin.*").strip()
        use_regex = True
    else:  # Field-specific
        search_field = st.selectbox("Search in field", SEARCHABLE_FIELDS)
        q = st.text_input(f"Search in {search_field}", placeholder=f"Search specifically in {search_field}").strip()
        use_regex = st.checkbox("Use regex", value=False)

    # Group filter
    all_groups = sorted({g for d in drugs[:1000] for g in extract_groups(d)})  # Sample for performance
    sel_groups = st.multiselect("Filter by group(s)", options=all_groups)

    # Additional filters
    st.subheader("Advanced Filters")
    has_targets = st.checkbox("Has targets")
    has_interactions = st.checkbox("Has drug interactions") 
    has_description = st.checkbox("Has description")
    
    # Display options
    st.subheader("Display Options")
    if search_mode == "Basic":
        use_regex = st.checkbox("Use regex search", value=False)
    show_raw = st.checkbox("Show raw JSON", value=False)
    show_detailed = st.checkbox("Show detailed fields", value=True)
    page_size = st.selectbox("Results per page", [10, 20, 50, 100], index=1)

# Enhanced filtering logic
def matches_query(drug: Dict[str, Any], text_blob: str, query: str, search_mode: str, search_field: str = None) -> bool:
    """Enhanced query matching with different search modes"""
    if not query:
        return True
    
    query_lower = query.lower()
    
    if search_mode == "Field-specific" and search_field:
        # Search in specific field only
        field_value = safe_get(drug, search_field).lower()
        if use_regex:
            try:
                return re.search(query, field_value, flags=re.IGNORECASE) is not None
            except re.error:
                return query_lower in field_value
        return query_lower in field_value
    else:
        # Search in full text blob
        if use_regex or search_mode == "Advanced":
            try:
                return re.search(query, text_blob, flags=re.IGNORECASE) is not None
            except re.error:
                return query_lower in text_blob
        return query_lower in text_blob

def matches_filters(drug: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if drug matches additional filters"""
    if filters.get("groups") and not matches_groups(drug, filters["groups"]):
        return False
    
    if filters.get("has_targets") and not drug.get("targets"):
        return False
        
    if filters.get("has_interactions") and not drug.get("drug-interactions"):
        return False
        
    if filters.get("has_description") and not drug.get("description"):
        return False
    
    return True

def matches_groups(drug: Dict[str, Any], want: List[str]) -> bool:
    if not want:
        return True
    gset = {g.lower() for g in extract_groups(drug)}
    return any(w.lower() in gset for w in want)

# Prepare filters
filters = {
    "groups": sel_groups,
    "has_targets": has_targets,
    "has_interactions": has_interactions,
    "has_description": has_description
}

# Log search activity
if q:
    logger.info(f"Search query: '{q}' (mode: {search_mode}, regex: {use_regex})")

# Compute results
start_time = datetime.now()
candidates: List[int] = []
for i, blob in enumerate(index_text):
    drug = drugs[i]
    if matches_query(drug, blob, q, search_mode, locals().get('search_field')) and matches_filters(drug, filters):
        candidates.append(i)

search_time = (datetime.now() - start_time).total_seconds()
logger.info(f"Search completed in {search_time:.3f}s, found {len(candidates)} results")

st.write(f"**{len(candidates)}** result(s) found in {search_time:.3f}s")

# Pagination
page = st.session_state.get("page", 1)
max_page = max(1, (len(candidates) + page_size - 1) // page_size)

col_a, col_b, col_c = st.columns([1, 2, 1], vertical_alignment="center")
with col_a:
    if st.button("⬅️ Prev", disabled=(page <= 1)):
        page = max(1, page - 1)
with col_c:
    if st.button("Next ➡️", disabled=(page >= max_page)):
        page = min(max_page, page + 1)
with col_b:
    page = st.number_input("Page", min_value=1, max_value=max_page, value=page, step=1, label_visibility="collapsed")

st.session_state["page"] = page

start = (page - 1) * page_size
end = start + page_size
page_idxs = candidates[start:end]

# Enhanced results display
for idx in page_idxs:
    d = drugs[idx]
    name = safe_get(d, "name") or "(no name)"
    ids = ", ".join(flatten_ids(d)) or "—"
    cas = safe_get(d, "cas-number") or "—"
    unii = safe_get(d, "unii") or "—"
    groups = ", ".join(extract_groups(d)) or "—"
    state = safe_get(d, "state") or "—"

    with st.container(border=True):
        # Header with key identifiers
        header_cols = st.columns([3, 2, 2, 2, 2])
        header_cols[0].markdown(f"### 💊 {name}")
        header_cols[1].markdown(f"**ID:** {ids}")
        header_cols[2].markdown(f"**CAS:** {cas}")
        header_cols[3].markdown(f"**UNII:** {unii}")
        header_cols[4].markdown(f"**State:** {state}")
        
        # Groups and categories in a separate row
        info_cols = st.columns([2, 3])
        info_cols[0].markdown(f"**Groups:** {groups}")
        categories = extract_categories(d)
        if categories:
            info_cols[1].markdown(f"**Categories:** {', '.join(categories[:3])}{'...' if len(categories) > 3 else ''}")

        # Core Information
        with st.expander("📋 Core Information", expanded=False):
            core_cols = st.columns(2)
            with core_cols[0]:
                for field in ["description", "indication", "mechanism-of-action", "pharmacodynamics"]:
                    val = safe_get(d, field)
                    if val:
                        st.markdown(f"**{field.replace('-', ' ').title()}:**")
                        # Truncate long text for display
                        display_text = val[:500] + "..." if len(val) > 500 else val
                        st.write(display_text)
                        
            with core_cols[1]:
                # Synonyms and identifiers
                syns = extract_synonyms(d)
                if syns:
                    st.markdown("**Synonyms & Brands:**")
                    st.write(", ".join(syns[:10]) + ("..." if len(syns) > 10 else ""))
                
                # Physical properties
                mass = safe_get(d, "average-mass")
                if mass:
                    st.markdown(f"**Average Mass:** {mass}")
                
                mono_mass = safe_get(d, "monoisotopic-mass")
                if mono_mass:
                    st.markdown(f"**Monoisotopic Mass:** {mono_mass}")

        # Detailed Fields (if enabled)
        if show_detailed:
            detail_tabs = st.tabs(["🎯 Targets", "⚠️ Interactions", "📊 Properties", "🧬 Pharmacology"])
            
            with detail_tabs[0]:
                targets_info = extract_detailed_field(d, "targets")
                st.markdown(f"**Summary:** {targets_info['summary']}")
                if targets_info['details']:
                    for target in targets_info['details'][:5]:  # Limit display
                        st.write(f"• **{target['name']}** ({target.get('organism', 'Unknown organism')})")
                        if target.get('actions'):
                            st.write(f"  Actions: {', '.join(target['actions'])}")
            
            with detail_tabs[1]:
                interactions_info = extract_detailed_field(d, "drug-interactions")
                st.markdown(f"**Summary:** {interactions_info['summary']}")
                if interactions_info['details']:
                    for interaction in interactions_info['details'][:5]:  # Limit display
                        st.write(f"• **{interaction['drug']}**")
                        if interaction.get('description'):
                            st.write(f"  {interaction['description'][:200]}...")
            
            with detail_tabs[2]:
                # Pharmacokinetic properties
                pk_fields = ["absorption", "metabolism", "half-life", "clearance", 
                           "protein-binding", "volume-of-distribution", "route-of-elimination"]
                for field in pk_fields:
                    val = safe_get(d, field)
                    if val:
                        st.markdown(f"**{field.replace('-', ' ').title()}:**")
                        display_text = val[:300] + "..." if len(val) > 300 else val
                        st.write(display_text)
            
            with detail_tabs[3]:
                # Toxicity and safety
                tox_fields = ["toxicity", "food-interactions", "affected-organisms"]
                for field in tox_fields:
                    val = safe_get(d, field)
                    if val:
                        st.markdown(f"**{field.replace('-', ' ').title()}:**")
                        display_text = val[:300] + "..." if len(val) > 300 else val
                        st.write(display_text)

        # All available sections
        with st.expander("🗂️ All Available Sections", expanded=False):
            available_fields = [k for k in sorted(d.keys()) if d.get(k)]
            missing_fields = [k for k in DETAILED_FIELDS if k not in d or not d.get(k)]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Available:** " + ", ".join(available_fields))
            with col2:
                if missing_fields:
                    st.markdown("**Missing:** " + ", ".join(missing_fields))

        # Raw JSON (if enabled)
        if show_raw:
            with st.expander("📄 Raw JSON Data", expanded=False):
                st.json(d)

# Footer with enhanced tips
st.divider()
st.markdown("### 💡 Tips & Usage")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Search Tips:**
    - Basic: Simple text search across all fields
    - Advanced: Use regex patterns (e.g., `(?i)aspirin|acetaminophen`)
    - Field-specific: Search within a specific field only
    - Use filters to narrow down by drug status, targets, etc.
    """)

with col2:
    st.markdown("""
    **Performance:**
    - Dataset contains {total_drugs} drugs
    - Search index built for fast queries
    - Detailed view shows comprehensive drug information
    - All operations logged to `drugbank_app.log`
    """.format(total_drugs=len(drugs)))

# Performance stats
if len(candidates) > 0:
    st.caption(f"Last search: {len(candidates)} results in {search_time:.3f}s")
