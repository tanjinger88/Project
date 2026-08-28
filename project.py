import streamlit as st
import feedparser
import requests
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="Institutional Market Intelligence | Enterprise Portal", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define API Key from Streamlit Secrets ---
API_KEY = st.secrets["API_KEY"]

# --- Enterprise Banking Dark Mode & OCBC Red Theme Styling ---
st.markdown("""
    <style>
    /* Global Application Canvas (Dark Mode) */
    .stApp {
        background-color: #0E1117;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Executive Headers */
    h1 {
        color: #DA291C !important;
        font-weight: 700;
        font-size: 2.3rem;
        letter-spacing: -0.5px;
        margin-bottom: 0rem;
    }
    
    h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600;
    }
    
    p, span, label, .stMarkdown {
        color: #E2E8F0 !important;
    }
    
    /* Subtitle Custom Styling */
    .banner-subtitle {
        color: #CBD5E1 !important;
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.5;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
    
    /* Professional Red-Hue KPI Metric Boxes */
    [data-testid="stMetric"] {
        background-color: #261515 !important;
        border: 1px solid #5C2626 !important;
        border-top: 4px solid #E63946 !important;
        border-radius: 6px;
        padding: 1.2rem !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        text-align: center !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        display: block !important;
    }
    [data-testid="stMetricLabel"] {
        color: #FF8080 !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        display: block !important;
    }
    
    /* Primary Action Buttons (Green Variant for Export) */
    .stDownloadButton button {
        background-color: #1E8E3E !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600;
        border-radius: 4px;
        padding: 0.5rem 1.25rem;
        box-shadow: 0 2px 4px rgba(30, 142, 62, 0.3);
        transition: background-color 0.2s;
    }
    .stDownloadButton button:hover {
        background-color: #176F31 !important;
    }
    
    /* Professional Regulatory Notice Styling */
    .sidebar-disclaimer {
        background-color: #1A1212;
        border-left: 4px solid #DA291C;
        border-top: 1px solid #3A2222;
        border-right: 1px solid #3A2222;
        border-bottom: 1px solid #3A2222;
        color: #FCA5A5;
        padding: 1.1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        line-height: 1.6;
        text-align: left;
        margin-top: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Expander Text Fix */
    .streamlit-expanderHeader {
        color: #E2E8F0 !important;
        background-color: #161B22 !important;
        border: 1px solid #30363D;
    }
    </style>
""", unsafe_allow_html=True)

MASTER_CSV = "master_market_archive.csv"
EXCEL_TRACKER = "download_audit_tracker.xlsx"

# --- Top Navigation / Banner Branding ---
st.markdown("<h1>Multi-Source Financial Market Recommendation Report</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="banner-subtitle">
An Overview of Financial Market News aggregated from Yahoo Finance, Investing.com, and Alpha Vantage. 
</div>
""", unsafe_allow_html=True)
st.markdown("---")

def get_selected_week_range(selected_date):
    """Takes any selected date and calculates the Monday-to-Sunday week window in MYT."""
    myt_zone = ZoneInfo("Asia/Kuala_Lumpur")
    target_date = selected_date[0] if isinstance(selected_date, tuple) else selected_date
    dt_myt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=myt_zone)
    monday_myt = dt_myt - timedelta(days=dt_myt.weekday())
    sunday_myt = monday_myt + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return monday_myt, sunday_myt, f"{monday_myt.strftime('%Y-%m-%d')} to {sunday_myt.strftime('%Y-%m-%d')}"

def parse_entry_date(entry):
    myt_zone = ZoneInfo("Asia/Kuala_Lumpur")
    time_struct = getattr(entry, 'published_parsed', None) or getattr(entry, 'updated_parsed', None)
    if time_struct:
        try:
            dt_utc = datetime(*time_struct[:6], tzinfo=ZoneInfo("UTC"))
            return dt_utc.astimezone(myt_zone)
        except Exception:
            pass
    return None

def get_yahoo_news(start_dt, end_dt):
    feed = feedparser.parse("https://finance.yahoo.com/news/rssindex")
    articles = []
    for entry in feed.entries:
        pub_date = parse_entry_date(entry)
        if pub_date and start_dt <= pub_date <= end_dt:
            articles.append({
                "title": entry.title, 
                "link": entry.link, 
                "source": "Yahoo Finance",
                "published_date": pub_date.strftime('%Y-%m-%d %H:%M:%S %Z')
            })
    return articles

def get_investing_news(start_dt, end_dt):
    feed = feedparser.parse("https://www.investing.com/rss/news_25.rss")
    articles = []
    for entry in feed.entries:
        pub_date = parse_entry_date(entry)
        if pub_date and start_dt <= pub_date <= end_dt:
            articles.append({
                "title": entry.title, 
                "link": entry.link, 
                "source": "Investing.com",
                "published_date": pub_date.strftime('%Y-%m-%d %H:%M:%S %Z')
            })
    return articles

def get_alpha_vantage_news(start_dt, end_dt, api_key=API_KEY):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=financial_markets&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        articles = []
        if "feed" in data:
            for item in data["feed"]:
                if item.get("title") and item.get("time_published"):
                    time_str = item.get("time_published")
                    dt_utc = datetime.strptime(time_str, "%Y%m%dT%H%M%S").replace(tzinfo=ZoneInfo("UTC"))
                    pub_date = dt_utc.astimezone(ZoneInfo("Asia/Kuala_Lumpur"))
                    if start_dt <= pub_date <= end_dt:
                        articles.append({
                            "title": item.get("title"), 
                            "link": item.get("url", "No URL provided"), 
                            "source": "Alpha Vantage",
                            "published_date": pub_date.strftime('%Y-%m-%d %H:%M:%S %Z')
                        })
        return articles
    except Exception:
        return []

def extract_live_justification(url):
    if not url or url == "No URL provided":
        return "No valid URL available for text extraction."
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        if text:
            sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 30]
            if sentences:
                return " ".join(sentences[:2])
            return text[:400] + "..."
    except Exception:
        pass
    return "Live text extraction unavailable."

def analyze_action(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ["drop", "fall", "slump", "crash", "inflation", "risk", "sell", "concern", "cut", "down", "loss", "tumble"]):
        return "SELL / CAUTION"
    elif any(word in text_lower for word in ["surge", "jump", "rally", "gain", "growth", "high", "buy", "record", "up", "beat", "upbeat", "doubles"]):
        return "BUY / ACCUMULATE"
    else:
        return "HOLD / MONITOR"

def find_common_and_recommend(yahoo, investing, av):
    all_articles = yahoo + investing + av
    if not all_articles:
        return []

    titles = [art["title"] for art in all_articles]
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = vectorizer.fit_transform(titles)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    matched_results = []
    processed_indices = set()
    SIMILARITY_THRESHOLD = 0.15
    
    for i in range(len(all_articles)):
        if i in processed_indices:
            continue
            
        group = [i]
        for j in range(i + 1, len(all_articles)):
            if j not in processed_indices and similarity_matrix[i, j] >= SIMILARITY_THRESHOLD:
                group.append(j)
                
        sources_in_group = set(all_articles[idx]["source"] for idx in group)
        
        if len(group) >= 2 and len(sources_in_group) >= 2:
            primary_art = all_articles[i]
            matched_links = []
            for idx in group:
                art = all_articles[idx]
                matched_links.append(f"[{art['source']}] (Published: {art['published_date']}) - {art['link']}")
                processed_indices.add(idx)
                
            action = analyze_action(primary_art["title"])
            live_justification = extract_live_justification(primary_art['link'])
            
            matched_results.append({
                "Primary Headline": primary_art['title'],
                "Published Date (MYT)": primary_art['published_date'],
                "Recommendation Action": action,
                "Live Justification": live_justification,
                "Sources Reporting": ", ".join(sources_in_group),
                "All Matched Links": " | ".join(matched_links)
            })

    return matched_results

def log_to_excel_tracker(df_to_log, week_str):
    myt_zone = ZoneInfo("Asia/Kuala_Lumpur")
    download_time = datetime.now(myt_zone).strftime('%Y-%m-%d %H:%M:%S %Z')
    df_download = df_to_log.copy()
    df_download["Downloaded Week"] = week_str
    df_download["Last Download Timestamp"] = download_time
    
    if os.path.exists(EXCEL_TRACKER):
        try:
            df_existing = pd.read_excel(EXCEL_TRACKER)
            df_combined = pd.concat([df_existing, df_download], ignore_index=True)
            df_cleaned = df_combined.drop_duplicates(subset=["Primary Headline", "Downloaded Week"], keep="last")
            df_cleaned.to_excel(EXCEL_TRACKER, index=False)
        except Exception:
            df_download.to_excel(EXCEL_TRACKER, index=False)
    else:
        df_download.to_excel(EXCEL_TRACKER, index=False)

# --- Load Master Archive CSV & Clean Data Older Than 1 Month ---
myt_zone = ZoneInfo("Asia/Kuala_Lumpur")
current_dt_myt = datetime.now(myt_zone)
one_month_ago_dt = current_dt_myt - timedelta(days=30)

if os.path.exists(MASTER_CSV):
    df_master = pd.read_csv(MASTER_CSV)
    if not df_master.empty and "Archived Week" in df_master.columns:
        df_master['Parsed_Week_Start'] = pd.to_datetime(df_master['Archived Week'].apply(lambda x: x.split(" to ")[0]))
        df_master = df_master[df_master['Parsed_Week_Start'] >= pd.to_datetime(one_month_ago_dt.strftime('%Y-%m-%d'))]
        df_master = df_master.drop(columns=['Parsed_Week_Start'])
        df_master.to_csv(MASTER_CSV, index=False)
else:
    df_master = pd.DataFrame(columns=["Primary Headline", "Published Date (MYT)", "Recommendation Action", "Live Justification", "Sources Reporting", "All Matched Links", "Archived Week"])

# --- AUTOMATED BACKGROUND SCRAPER FOR CURRENT WEEK ---
current_start_dt, current_end_dt, current_target_str = get_selected_week_range(current_dt_myt.date())

current_week_exists = False
if not df_master.empty and "Archived Week" in df_master.columns:
    current_week_exists = current_target_str in df_master["Archived Week"].values

if not current_week_exists:
    with st.spinner("Executing background feeds synchronization & vector cross-matching..."):
        yahoo_headlines = get_yahoo_news(current_start_dt, current_end_dt)
        investing_headlines = get_investing_news(current_start_dt, current_end_dt)
        av_headlines = get_alpha_vantage_news(current_start_dt, current_end_dt, API_KEY) 
        
        new_results = find_common_and_recommend(yahoo_headlines, investing_headlines, av_headlines)
        
        if new_results:
            df_new = pd.DataFrame(new_results)
            df_new["Archived Week"] = current_target_str
            
            if not df_master.empty:
                df_master = pd.concat([df_master, df_new]).drop_duplicates(subset=["Primary Headline", "Archived Week"], keep="last")
            else:
                df_master = df_new
                
            df_master.to_csv(MASTER_CSV, index=False)

available_weeks = []
if not df_master.empty and "Archived Week" in df_master.columns:
    available_weeks = sorted(df_master["Archived Week"].dropna().unique().tolist(), reverse=True)

# --- Sidebar Controls ---
st.sidebar.markdown("### ⚙️ Timeline Settings")

if available_weeks:
    target_date_str = st.sidebar.selectbox(
        "Reporting Period (Mon - Sun):",
        options=available_weeks,
        help="Access archived weekly summaries from the past 30 days."
    )
    selected_date = datetime.strptime(target_date_str.split(" to ")[0], "%Y-%m-%d").date()
    start_dt, end_dt, _ = get_selected_week_range(selected_date)
else:
    st.sidebar.warning("No archived reports found.")
    start_dt, end_dt, target_date_str = get_selected_week_range(current_dt_myt.date())

st.sidebar.markdown("---")
st.sidebar.info(f"**Selected Audit Window:**\n{target_date_str}")

# --- Push Regulatory Notice to the Bottom of Sidebar ---
st.sidebar.markdown("<br>" * 10, unsafe_allow_html=True) 
st.sidebar.markdown(
    """<div class="sidebar-disclaimer">
    <strong>⚠️ Regulatory Notice</strong><br><br>
    This document is generated for institutional and internal evaluation only. The recommendations provided are indicative, derived from automated multi-source synthesis, and do not constitute formal financial advice or binding portfolio commitments.
    </div>""",
    unsafe_allow_html=True
)

# --- Filter Data for Selected Week ---
filtered_df = df_master[df_master["Archived Week"] == target_date_str] if (not df_master.empty and "Archived Week" in df_master.columns) else pd.DataFrame()

if not filtered_df.empty:
    # --- Executive KPI Summary Metrics in Styled Red Boxes ---
    col1, col2, col3, col4 = st.columns(4)
    total_recs = len(filtered_df)
    buy_count = len(filtered_df[filtered_df["Recommendation Action"].str.contains("BUY", na=False)])
    sell_count = len(filtered_df[filtered_df["Recommendation Action"].str.contains("SELL", na=False)])
    hold_count = total_recs - (buy_count + sell_count)
    
    with col1:
        st.metric(label="Total Top Headlines", value=total_recs)
    with col2:
        st.metric(label="Buy / Accumulate Signals", value=buy_count)
    with col3:
        st.metric(label="Sell / Caution Signals", value=sell_count)
    with col4:
        st.metric(label="Hold / Monitor Signals", value=hold_count)
        
    st.markdown("---")

    # --- Header Action Toolbar ---
    col_head1, col_head2 = st.columns([5.5, 0.8])
    with col_head1:
        st.subheader("Weekly Summarisation")
    with col_head2:
        csv_bytes = filtered_df.drop(columns=["Archived Week"]).to_csv(index=False).encode('utf-8')
        st.markdown("<div style='display: flex; justify-content: flex-end;'>", unsafe_allow_html=True)
        st.download_button(
            label="Export CSV Report",
            data=csv_bytes,
            file_name=f"market_intelligence_report_{target_date_str.replace(' ', '_')}.csv",
            mime="text/csv",
            on_click=log_to_excel_tracker,       
            args=(filtered_df, target_date_str)    
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("")

    # --- Recommendation Cards Display ---
    for idx, row in filtered_df.reset_index(drop=True).iterrows():
        action_tag = row['Recommendation Action']
        
        badge_color = "#1E8E3E" if "BUY" in action_tag else ("#D93025" if "SELL" in action_tag else "#F29900")
        formatted_date = str(row['Published Date (MYT)']).split()[0]
        
        with st.container():
            st.markdown(f"""
                <div style="background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                        <span style="font-size: 0.85rem; font-weight: 600; color: #8B949E;">HEADLINE #{idx + 1} &bull; PUBLISHED: {formatted_date}</span>
                        <span style="background-color: {badge_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;">{action_tag}</span>
                    </div>
                    <h2 style="margin-top: 0rem; margin-bottom: 1rem; font-size: 1.5rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.3px;">{row['Primary Headline']}</h2>
                    <div style="margin-bottom: 0.75rem; color: #C9D1D9; font-size: 0.95rem;">
                        <strong style="color: #F8FAFC;">Analysis & Justification:</strong><br>
                        <div style="margin-top: 0.35rem; line-height: 1.5;">{row['Live Justification']}</div>
                    </div>
                    <p style="margin-bottom: 0.5rem; color: #8B949E; font-size: 0.85rem;"><strong>Sources:</strong> {row['Sources Reporting']}</p>
            """, unsafe_allow_html=True)
            
            with st.expander("🔗 View Cross-Referenced Source Links"):
                for link in str(row['All Matched Links']).split(" | "):
                    st.markdown(f"- {link}")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
else:
    st.info(f"No records available inside `master_market_archive.csv` for the selected reporting window: **{target_date_str}**.")