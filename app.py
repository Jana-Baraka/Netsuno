from curses.ascii import alt
import streamlit as st
import geopandas as gpd
import folium
from core.network_design import NetworkDesigner
from core.resource_alloc import ResourceOptimizer
from core.policy_analyzer import ProcurementAnalyzer
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
designer = NetworkDesigner()
optimizer = ResourceOptimizer()
procurement = ProcurementAnalyzer()

# Track 1: Network Planning
with st.expander("📡 Network Design"):
    col1, col2 = st.columns([3,1])
    with col1:
        schools = designer.load_schools_data()
        towers = gpd.read_file("https://opencellid.org/ocid/downloads")
        gaps = designer.analyze_coverage_gaps(schools, towers)
        
        m = folium.Map()
        folium.GeoJson(gaps).add_to(m)
        st_folium(m, width=800)
    
    with col2:
        st.metric("Unconnected Schools", len(gaps))
        st.download_button("Export CSV", gaps.to_csv())

# Track 2: Resource Allocation
with st.expander("⚡ Performance Optimization"):
    outages = optimizer.get_outage_alerts()
    priority_scores = [optimizer.calculate_priority_score(s, outages) 
                      for s in schools]
    
    st.altair_chart(alt.Chart(schools).mark_circle().encode(
        x='longitude',
        y='latitude',
        size='priority_score'
    ))

# Track 3: Procurement Analysis 
with st.expander("📜 Contract Analysis"):
    contract_text = st.text_area("Paste contract")
    if contract_text:
        risk_score = procurement.score_vendor_risk(contract_text)
        st.progress(risk_score / 100)
        st.write(f"Vendor Risk Score: {risk_score:.1f}/100")