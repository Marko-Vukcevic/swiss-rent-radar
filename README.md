# Swiss Rent Radar

## Project: Drivers of Swiss Rental Prices — Combining Official BFS Statistics with Live Listings

### Group Members
- Marko Vukcevic
- Aladin Kermo

### Objective
To investigate which structural and geographic factors most strongly explain
rent differences across Switzerland — and to quantify how current asking
prices on the open market deviate from the official rent index published
by the Federal Statistical Office (BFS). The analysis combines two
independent data sources, applies three statistical tests, and presents
the results in an interactive Streamlit web application.

### Research Question
> *Which canton- and apartment-level factors best explain monthly rent in
> Switzerland, and where do live Homegate asking prices systematically
> diverge from the BFS reference?*

### Data Sources
- [opendata.swiss / Federal Statistical Office (BFS)](https://opendata.swiss/) — official canton-level rent statistics (mean rent per room class, mean price per m²)
- [Homegate.ch](https://www.homegate.ch) — live apartment listings, scraped at low volume for academic research
- [Swiss canton GeoJSON](https://github.com/interactivethings/swiss-maps) — boundaries for the choropleth map

### How to Run the Project
1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/<your-user>/swiss-rent-radar.git
   cd swiss-rent-radar
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. *(Optional)* copy `.env.example` to `.env` and add a free Together.ai
   API key for the LLM-powered market commentary.
3. Run the end-to-end data pipeline (collects + cleans + writes SQLite):
   ```bash
   python scripts/run_pipeline.py
   ```
4. Launch the interactive Streamlit web app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
5. Or explore the analysis step-by-step in the notebooks:
   `notebooks/01_data_collection.ipynb` →
   `02_data_preparation.ipynb` →
   `03_analysis_visualization.ipynb`.

### Project Highlights
- 🌐 **Web scraper**: Homegate.ch listings parsed from the embedded `__NEXT_DATA__` JSON with BeautifulSoup
- 🔌 **REST API**: opendata.swiss / BFS rent statistics
- 🧹 **Data preparation**: regular expressions parse Swiss-formatted strings (`CHF 2'500.–`, `3.5 Zimmer`, `85 m²`)
- 💾 **Database**: SQLite with `JOIN` / `GROUP BY` SQL queries
- 🧱 **OOP**: `Listing` and `RentalMarket` classes alongside the procedural pipeline
- 📈 **Statistics**: Pearson correlation, Welch's *t*-test, one-way ANOVA — all reporting p-values
- 📊 **Visualisations**: matplotlib + seaborn charts, folium interactive map, choropleth
- 🤖 **LLM**: Together.ai (Llama 3.3 70B) generates plain-language market commentary
- 🖥️ **Web app**: Streamlit dashboard with sidebar filters, live SQL, charts, map and LLM button
- 🔄 **GitHub**: full code, notebooks, and pipeline available publicly (large data excluded via `.gitignore`)

### Repository Structure
```
swiss-rent-radar/
├── app/                # Python package: scraper, cleaning, DB, analysis, viz, LLM, web app, OOP
├── notebooks/          # 01_data_collection · 02_data_preparation · 03_analysis_visualization
├── scripts/            # run_pipeline.py — end-to-end orchestrator
├── data/               # CSVs and SQLite DB (gitignored)
├── outputs/            # Generated maps and figures (gitignored)
├── presentation/       # Final video (mp4) and slides (pdf) per submission rubric
├── requirements.txt
├── .gitignore
├── .env.example
├── LICENSE
└── README.md
```
