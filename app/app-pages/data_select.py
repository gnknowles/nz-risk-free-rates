import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# --------------------
# Load Data
# --------------------
rbnz_file = os.path.abspath("../data/003_rbnz_yields_transformed.csv")

try:
    df_rbnz = pd.read_csv(rbnz_file, parse_dates=["date"])
except Exception as e:
    st.error(f"Failed to load RBNZ file: {e}")
    st.stop()

# --------------------
# Set page layout (wider)
# --------------------
st.set_page_config(layout="wide")
st.title("🎯 Select Assets to include in Curve")

# --------------------
# 1. Date Filter
# --------------------
available_dates = sorted(df_rbnz["date"].dt.date.unique())
if not available_dates:
    st.warning("No valuation dates found.")
    st.stop()

selected_date = st.date_input(
    "Select Valuation Date:",
    value=max(available_dates),
    min_value=min(available_dates),
    max_value=max(available_dates),
)

if selected_date not in available_dates:
    st.warning("No data available for the selected date.")
    st.stop()

df_filtered = df_rbnz[df_rbnz["date"].dt.date == selected_date]

# --------------------
# 2. Group Filter
# --------------------
group_options = sorted(df_filtered["group"].dropna().unique())
selected_groups = st.multiselect("Filter by Group(s):", group_options, default=group_options)

df_filtered = df_filtered[df_filtered["group"].isin(selected_groups)] if selected_groups else df_filtered

# --------------------
# 3. Series Filter
# --------------------
series_options = sorted(df_filtered["series"].dropna().unique())
selected_series = st.multiselect("Filter by Series:", series_options, default=series_options)

df_filtered = df_filtered[df_filtered["series"].isin(selected_series)] if selected_series else df_filtered

# --------------------
# 4. Series ID Filter
# --------------------
series_label_map = {
    row["series_id"]: f"{row['series_id']} — {row['series']}"
    for _, row in df_filtered.drop_duplicates(subset="series_id").iterrows()
}

available_series_ids = list(series_label_map.keys())

selected_series_ids = st.multiselect(
    "Select Asset(s) (Series IDs):",
    options=available_series_ids,
    default=available_series_ids,
    format_func=lambda sid: series_label_map[sid]
)

# --------------------
# Confirm Selection
# --------------------
if st.button("✅ Confirm Selection"):
    if not selected_series_ids:
        st.warning("Please select at least one asset.")
    else:
        st.success(f"{len(selected_series_ids)} asset(s) selected.")
        st.session_state["selected_series_ids"] = selected_series_ids

# --------------------
# Final Filtered DataFrame
# --------------------
result_df = df_filtered[df_filtered["series_id"].isin(selected_series_ids)] if selected_series_ids else pd.DataFrame()

# --------------------
# Show Table
# --------------------
if not result_df.empty:
    st.markdown("### 📄 Filtered Data Table")
    st.dataframe(result_df, use_container_width=True)

# --------------------
# Plot Yield Curve
# --------------------
if not result_df.empty and all(col in result_df.columns for col in ["term_yr", "yield_decimal"]):
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
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)
elif selected_series_ids:
    st.warning("Missing 'term_yr' or 'yield_decimal' in the selected data.")

# --------------------
# Load and Filter NZDM Dataset
# --------------------
nzdm_file = os.path.abspath("../data/002_nzdm_govtbonds_onissue_enriched.csv")

try:
    df_nzdm = pd.read_csv(nzdm_file, parse_dates=["as_of_date", "maturity"])
except Exception as e:
    st.error(f"Failed to load NZDM data: {e}")
    df_nzdm = pd.DataFrame()

if not df_nzdm.empty and selected_series_ids:

    st.markdown("---")
    st.subheader("🏛️ NZDM Market Bonds")

    # 1️⃣ Filter by series_id first
    df_nzdm = df_nzdm[df_nzdm["series_id"].isin(selected_series_ids)]

    # 2️⃣ Get available as_of_dates
    available_nzdm_dates = sorted(df_nzdm["as_of_date"].dt.date.unique())
    if not available_nzdm_dates:
        st.info("No available NZDM dates for the selected series.")
    else:
        selected_nzdm_date = st.date_input(
            "Select NZDM Valuation Date:",
            value=max(available_nzdm_dates),
            min_value=min(available_nzdm_dates),
            max_value=max(available_nzdm_dates),
            key="nzdm_date_input"
        )

        # 3️⃣ Filter by selected as_of_date
        df_nzdm_filtered = df_nzdm[df_nzdm["as_of_date"].dt.date == selected_nzdm_date]

        if not df_nzdm_filtered.empty:
            # 4️⃣ Calculate term_yr
            df_nzdm_filtered["term_yr"] = (
                (df_nzdm_filtered["maturity"] - pd.to_datetime(selected_nzdm_date)).dt.days / 365.25
            )

            # 5️⃣ Show table
            st.markdown("### 🧾 Filtered NZDM Table")
            st.dataframe(df_nzdm_filtered, use_container_width=True)

            # 6️⃣ Plot term_yr vs market_bonds_m
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
                    margin=dict(l=40, r=40, t=60, b=40),
                    height=450
                )

                st.plotly_chart(fig_nzdm, use_container_width=True)
            else:
                st.warning("Required columns missing for NZDM plot.")
        else:
            st.info("No NZDM data for selected date.")
else:
    st.info("NZDM data not available or no matching series selected.")
