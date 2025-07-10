import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# --------------------
# Load Data
# --------------------
rbnz_file = os.path.abspath("../data/003_rbnz_yields_transformed.csv")
nzdm_file = os.path.abspath("../data/002_nzdm_govtbonds_onissue_enriched.csv")

try:
    df_rbnz = pd.read_csv(rbnz_file, parse_dates=["date"])
except Exception as e:
    st.error(f"Failed to load RBNZ data: {e}")
    st.stop()

try:
    df_nzdm = pd.read_csv(nzdm_file, parse_dates=["as_of_date", "maturity"])
except Exception as e:
    st.warning(f"NZDM data failed to load: {e}")
    df_nzdm = pd.DataFrame()

# --------------------
# Page Setup
# --------------------
st.set_page_config(layout="wide")
st.title("Select assets to use in discount curve")

# --------------------
# 1. RBNZ Date Filter
# --------------------
available_dates = sorted(df_rbnz["date"].dt.date.unique())
selected_date = st.date_input(
    "📅 Select RBNZ Valuation Date:",
    value=max(available_dates),
    min_value=min(available_dates),
    max_value=max(available_dates),
)

df_rbnz_filtered = df_rbnz[df_rbnz["date"].dt.date == selected_date]

if df_rbnz_filtered.empty:
    st.warning("No RBNZ data available for the selected date.")
    st.stop()

# --------------------
# 2. Group & Series Filters
# --------------------
group_options = sorted(df_rbnz_filtered["group"].dropna().unique())
selected_groups = st.multiselect("🔹 Filter by Group(s):", group_options, default=group_options)
df_rbnz_filtered = df_rbnz_filtered[df_rbnz_filtered["group"].isin(selected_groups)]

series_options = sorted(df_rbnz_filtered["series"].dropna().unique())
selected_series = st.multiselect("🔹 Filter by Series:", series_options, default=series_options)
df_rbnz_filtered = df_rbnz_filtered[df_rbnz_filtered["series"].isin(selected_series)]

# --------------------
# 3. Asset Selection
# --------------------
series_label_map = {
    row["series_id"]: f"{row['series_id']} — {row['series']}"
    for _, row in df_rbnz_filtered.drop_duplicates(subset="series_id").iterrows()
}

selected_series_ids = st.multiselect(
    "📌 Select Asset(s) (Series IDs):",
    options=list(series_label_map.keys()),
    default=list(series_label_map.keys()),
    format_func=lambda sid: series_label_map[sid],
)

# --------------------
# 4. Confirm Selection
# --------------------
confirmed = False
if st.button("✅ Confirm Selection"):
    if not selected_series_ids:
        st.warning("Please select at least one asset.")
    else:
        confirmed = True
        st.success(f"{len(selected_series_ids)} asset(s) selected.")

# --------------------
# 5. Display RBNZ Table and Plot
# --------------------
if confirmed and selected_series_ids:
    result_df = df_rbnz_filtered[df_rbnz_filtered["series_id"].isin(selected_series_ids)]

    st.markdown("### 📄 Filtered RBNZ Data")
    st.dataframe(result_df, use_container_width=True)

    if all(col in result_df.columns for col in ["term_yr", "yield_decimal"]):
        fig = go.Figure()
        for sid in selected_series_ids:
            df_plot = result_df[result_df["series_id"] == sid]
            fig.add_trace(go.Scatter(
                x=df_plot["term_yr"],
                y=df_plot["yield_decimal"],
                mode="markers+lines",
                name=series_label_map[sid],
                marker=dict(size=8)
            ))

        fig.update_layout(
            title=f"📈 Yield Curve — {selected_date.strftime('%d %B %Y')}",
            xaxis_title="Term (Years)",
            yaxis_title="Yield (Decimal)",
            height=450,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing 'term_yr' or 'yield_decimal' columns in RBNZ data.")

# --------------------
# 6. NZDM Integration
# --------------------
if confirmed and not df_nzdm.empty and selected_series_ids:
    st.markdown("---")
    st.subheader("🏛️ NZDM Market Bonds")

    df_nzdm_filtered = df_nzdm[df_nzdm["series_id"].isin(selected_series_ids)]

    available_nzdm_dates = sorted(df_nzdm_filtered["as_of_date"].dt.date.unique())
    if not available_nzdm_dates:
        st.info("No NZDM data available for selected assets.")
    else:
        selected_nzdm_date = st.date_input(
            "📅 Select NZDM Valuation Date:",
            value=max(available_nzdm_dates),
            min_value=min(available_nzdm_dates),
            max_value=max(available_nzdm_dates),
            key="nzdm_date_input"
        )

        df_nzdm_filtered = df_nzdm_filtered[df_nzdm_filtered["as_of_date"].dt.date == selected_nzdm_date]

        if not df_nzdm_filtered.empty:
            df_nzdm_filtered["term_yr"] = (
                (df_nzdm_filtered["maturity"] - pd.to_datetime(selected_nzdm_date)).dt.days / 365.25
            )

            st.markdown("### 🧾 Filtered NZDM Data")
            st.dataframe(df_nzdm_filtered, use_container_width=True)

            if all(col in df_nzdm_filtered.columns for col in ["term_yr", "market_bonds_m"]):
                fig_nzdm = go.Figure()

                for sid in selected_series_ids:
                    df_plot = df_nzdm_filtered[df_nzdm_filtered["series_id"] == sid]
                    fig_nzdm.add_trace(go.Bar(
                        x=df_plot["term_yr"],
                        y=df_plot["market_bonds_m"],
                        name=f"{sid} (NZDM)"
                    ))

                fig_nzdm.update_layout(
                    title=f"📊 NZDM Market Bonds — {selected_nzdm_date.strftime('%d %B %Y')}",
                    xaxis_title="Term (Years)",
                    yaxis_title="Market Bonds ($M)",
                    height=450,
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig_nzdm, use_container_width=True)
            else:
                st.warning("Missing required columns for NZDM plotting.")
        else:
            st.info("No NZDM data available for the selected date.")
elif confirmed:
    st.info("NZDM dataset not available or no matching data for selected series.")
