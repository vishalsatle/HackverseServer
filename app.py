import json

import streamlit as st
from dotenv import load_dotenv

from groq_agent_pipeline import (
    fetch_disaster_intel,
    fetch_live_osint_news,
    fetch_weather_intel,
    run_intel_pipeline,
)

load_dotenv()

st.set_page_config(
    page_title="Hackverse Intelligence Console",
    page_icon="🛰️",
    layout="wide",
)

st.title("Hackverse Multi-Source Intelligence Dashboard")
st.caption("CrewAI + Groq tactical brief generation with source traceability.")

if "results" not in st.session_state:
    st.session_state.results = {
        "conflict": {"data": None, "brief": None},
        "weather": {"data": None, "brief": None},
        "disaster": {"data": None, "brief": None},
    }

conflict_tab, weather_tab, disaster_tab = st.tabs(
    ["🌍 Conflict Intelligence", "⛈️ Weather Intelligence", "🌋 Disaster Intelligence"]
)


def run_and_render(tab_key: str, intel_data: list[dict[str, str]]) -> None:
    with st.spinner("Running CrewAI analysis..."):
        try:
            brief = run_intel_pipeline(intel_data)
            st.session_state.results[tab_key]["data"] = intel_data
            st.session_state.results[tab_key]["brief"] = brief
        except Exception as exc:
            st.error(f"Pipeline run failed: {exc}")


with conflict_tab:
    st.subheader("Live Conflict Feed (GNews)")
    if st.button("Run Analysis", key="run_conflict"):
        conflict_data = fetch_live_osint_news(top_n=5)
        run_and_render("conflict", conflict_data)

    if st.session_state.results["conflict"]["data"]:
        st.markdown("#### Ingested Intelligence")
        st.json(st.session_state.results["conflict"]["data"])
        st.markdown("#### Chief of Staff Brief")
        st.markdown(st.session_state.results["conflict"]["brief"])


with weather_tab:
    st.subheader("Live Weather Feed (OpenWeatherMap)")
    weather_location = st.text_input(
        "City / Region", value="Kyiv", key="weather_location"
    )
    if st.button("Run Weather Analysis", key="run_weather"):
        weather_data = fetch_weather_intel(location=weather_location)
        run_and_render("weather", weather_data)

    if st.session_state.results["weather"]["data"]:
        st.markdown("#### Ingested Intelligence")
        st.json(st.session_state.results["weather"]["data"])
        st.markdown("#### Chief of Staff Brief")
        st.markdown(st.session_state.results["weather"]["brief"])


with disaster_tab:
    st.subheader("Live Disaster Feed (ReliefWeb)")
    if st.button("Run Analysis", key="run_disaster"):
        disaster_data = fetch_disaster_intel(top_n=5)
        run_and_render("disaster", disaster_data)

    if st.session_state.results["disaster"]["data"]:
        st.markdown("#### Ingested Intelligence")
        st.json(st.session_state.results["disaster"]["data"])
        st.markdown("#### Chief of Staff Brief")
        st.markdown(st.session_state.results["disaster"]["brief"])


with st.expander("Debug: Current Session State"):
    st.code(json.dumps(st.session_state.results, indent=2), language="json")
