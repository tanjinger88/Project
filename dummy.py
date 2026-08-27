import pandas as pd
from datetime import datetime

# Let's generate dummy data for August 2026 weeks following the structure of master_market_archive.csv
# Columns: Primary Headline, Published Date (MYT), Recommendation Action, Live Justification, Sources Reporting, All Matched Links, Archived Week

dummy_rows = [
    {
        "Primary Headline": "Global Tech Stocks Rally as Inflation Cools Down",
        "Published Date (MYT)": "2026-08-05 10:30:00 MYT",
        "Recommendation Action": "BUY / ACCUMULATE",
        "Live Justification": "Global equity markets experienced a major boost following reports showing inflation easing below target expectations. Central banks are expected to pause rate hikes.",
        "Sources Reporting": "Yahoo Finance, Investing.com",
        "All Matched Links": "[Yahoo Finance] (Published: 2026-08-05 10:30:00 MYT) - https://finance.yahoo.com/news/sample1 | [Investing.com] (Published: 2026-08-05 09:15:00 MYT) - https://www.investing.com/news/sample1",
        "Archived Week": "2026-08-03 to 2026-08-09"
    },
    {
        "Primary Headline": "Energy Sector Faces Turbulence Amid Supply Chain Disruptions",
        "Published Date (MYT)": "2026-08-12 14:00:00 MYT",
        "Recommendation Action": "SELL / CAUTION",
        "Live Justification": "Crude oil prices tumbled as unexpected inventory surges combined with geopolitical friction led analysts to downgrade short-term energy sector outlooks.",
        "Sources Reporting": "Investing.com, Alpha Vantage",
        "All Matched Links": "[Investing.com] (Published: 2026-08-12 14:00:00 MYT) - https://www.investing.com/news/sample2 | [Alpha Vantage] (Published: 2026-08-12 13:30:00 MYT) - https://www.alphavantage.co/news/sample2",
        "Archived Week": "2026-08-10 to 2026-08-16"
    },
    {
        "Primary Headline": "Semiconductor Demand Surges on AI Expansion",
        "Published Date (MYT)": "2026-08-19 11:20:00 MYT",
        "Recommendation Action": "BUY / ACCUMULATE",
        "Live Justification": "Leading chipmakers reported record quarterly earnings driven by surging enterprise demand for artificial intelligence infrastructure and data centers.",
        "Sources Reporting": "Yahoo Finance, Alpha Vantage",
        "All Matched Links": "[Yahoo Finance] (Published: 2026-08-19 11:20:00 MYT) - https://finance.yahoo.com/news/sample3 | [Alpha Vantage] (Published: 2026-08-19 10:00:00 MYT) - https://www.alphavantage.co/news/sample3",
        "Archived Week": "2026-08-17 to 2026-08-23"
    }

]

df_dummy = pd.DataFrame(dummy_rows)
df_dummy.to_csv("master_market_archive.csv", index=False)
print("Successfully generated dummy data for August 2026.")