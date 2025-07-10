import streamlit as st

# Updated CSS for Treasury-style header (light blue)
welcome_css = '''
    color: white; 
    font-size: 28px; 
    text-align: center; 
    border-radius: 1rem; 
    padding: 1rem; 
    font-weight: bold; 
    background-color: #007A9C;  /* Treasury blue */
'''

st.markdown(f"<div style='{welcome_css}'>New Zealand Risk-free Discount Rate Tool</div>", unsafe_allow_html=True)

st.divider()

st.subheader("📌 Introduction")
st.markdown("""
Welcome to the **Risk-free Discount Rate Tool**, developed in alignment with [New Zealand Treasury guidance](https://www.treasury.govt.nz/information-and-services/public-sector-leadership/guidance/reporting-financial/discount-rates/discount-rates-and-cpi-assumptions-accounting-valuation-purposes).

This tool assists in generating risk-free discount rate curves based on current market data and selected parameters, based on Treasury-published methodologies. 

The risk-free discount rate methodology uses at its starting point the market yield curve of New Zealand Government Bonds as the most appropriate proxy for the return on a very safe asset.            
""")

st.subheader("📘 User Guide")
st.markdown("""
- 📅 Select a valuation date for the market data to be used (from RBNZ/NZDM website) and select the bills/bonds to use in the construction of the curve    
- 🛠️ Configure key parameters assumptions, including long-term discount rate
- 📊 Generate and visualize the forward rate curves, including selection of appropriate knots
- 💾 Export results as CSV download.

Refer to the [official Treasury page](https://www.treasury.govt.nz/information-and-services/public-sector-leadership/guidance/reporting-financial/discount-rates) for full documentation and latest assumptions.
""")

st.divider()

if st.button("🚀 Launch Tool", type="primary"):
    st.switch_page("app-pages/page1.py")
