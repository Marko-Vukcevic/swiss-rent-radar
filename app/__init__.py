"""
swiss-rent-radar
================

A Scientific Programming project (FS2026, ZHAW) analysing the Swiss rental
market by combining official statistics from opendata.swiss with live
listings scraped from Homegate.

Modules
-------
- api_client      : opendata.swiss / BFS REST client
- scraper         : Homegate web scraper (BeautifulSoup)
- data_cleaning   : regex-based parsing and pandas cleaning
- database        : SQLite schema, inserts, SQL queries
- analysis        : Pearson correlation, ANOVA, Welch t-test
- visualization   : matplotlib + folium choropleth
- llm_helper      : Together.ai market commentary
- models          : Listing and RentalMarket OOP classes
- streamlit_app   : interactive web dashboard
"""

__version__ = "1.0.0"
__author__ = "Marko Vukcevic and contributors"
