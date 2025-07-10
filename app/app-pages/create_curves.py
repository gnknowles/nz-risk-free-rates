import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import io
import base64
import plotly.graph_objects as go
from openai import OpenAI
import plotly.io as pio
import re

# Add your local project path for utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import ModellingFunctions


# Load and validate session state data
def load_data_from_session():
    required_keys = ["rbnz_filtered", "nzdm_filtered"]
    missing_keys = [k for k in required_keys if k not in st.session_state]

    if missing_keys:
        st.error(f"Missing required data: {', '.join(missing_keys)}. Please return to the selection page.")
        return None, None, None

    pdf_yields = st.session_state["rbnz_filtered"]
    pdf_amounts = st.session_state["nzdm_filtered"]
    as_of_date = st.session_state.get("selected_rbnz_date", None)

    return pdf_yields, pdf_amounts, as_of_date


# Bootstrap forward curve
def bootstrap_forward_curve(merged_pdf):
    df_bootstrapped = ModellingFunctions.bootstrap_forward_columns(
        merged_pdf,
        term_col="term_yr",
        spot_col="yield_decimal"
    )
    return df_bootstrapped


# Plot the bootstrapped forward curve and market yields
import plotly.graph_objects as go

def plot_forward_curve(
    df,
    x_col: str = "term_yr",
    line1_col: str = None,
    line1_name: str = "Line 1",
    line2_col: str = None,
    line2_name: str = "Line 2",
    scatter_col: str = None,
    scatter_name: str = "Market Yields",
    show_step_lines: bool = True
):
    fig = go.Figure()

    # Add first line trace
    if line1_col and line1_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[line1_col],
            mode="lines",
            name=line1_name,
            line_shape="hv" if show_step_lines else "linear",
            line=dict(width=2)
        ))

    # Add second line trace
    if line2_col and line2_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[line2_col],
            mode="lines",
            name=line2_name,
            line_shape="linear",  # force linear for fitted spline or make optional
            line=dict(width=2, dash="dot")
        ))

    # Add scatter trace
    if scatter_col and scatter_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[scatter_col],
            mode="markers",
            name=scatter_name,
            marker=dict(size=8, symbol="circle", color="black")
        ))

    fig.update_layout(
        title="Forward Curve Plot",
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title="Rate",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )

    return fig


# Ask ChatGPT for knot placement advice using vision
def ask_chatgpt_for_knot_advice(fig, df_bootstrapped):
    try:
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial modeling assistant helping design cubic spline interpolations on forward rate curves."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":
                            "Based on the chart and table below, suggest exactly 4 knot positions (in years) for cubic spline interpolation. "
                            "Consider the weights in the table, and ensure one of the 4 knots is toward the table to control the long end of the forward curve."
                            "Also would expect one of the knots to be <1"
                            "If needed, you can choose specific decimal values to best capture inflection points and slope changes.\n\n"
                            "Return only:\n"
                            "- A list of 4 decimal knot values in ascending order (e.g., [1.0, 4.75, 12.5, 26.0])\n"
                            "- One sentence explaining why each knot was chosen.\n"
                            "- A short explanation overall on the selection of knots and any considerations or judgements.\n\n"
                            + format_curve_table(df_bootstrapped)
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"ChatGPT request failed: {e}")
        return None

def format_curve_table(df):
    subset = df[["term_yr", "fwd_rate_bootstrapped"]].round(4)
    table_text = "Forward Curve Data:\n"
    table_text += "Term (Years) | Fwd Rate\n"
    table_text += "-------------|-----------\n"
    for _, row in subset.iterrows():
        table_text += f"{row['term_yr']:>12.2f} | {row['fwd_rate_bootstrapped']:.4f}\n"
    return table_text


def extract_knot_list(response_text: str):
    """
    Extracts a list of 4 float values from a string like: '[2.0, 6.5, 12.0, 25.0]'
    Returns an empty list if not found.
    """
    match = re.search(r"\[(.*?)\]", response_text)
    if match:
        try:
            return [float(k.strip()) for k in match.group(1).split(",")]
        except ValueError:
            return []
    return []


# Fit Cubic Spline
def fit_cubic_spline_forward(merged_pdf, knots, df_bootstrapped):

    df = merged_pdf.sort_values(by='term_yr')[["term_yr", "yield_decimal", "market_bonds_m"]].rename(columns={"yield_decimal": "spot_rate_pa"})

    # Set a default high weight to T-bills and OCR with no weight
    df["weight"] = np.where(
        df["market_bonds_m"].isnull(),
        4e9,
        np.minimum(df["market_bonds_m"], 4e9)
    )

    spline, fitted_rates, res = ModellingFunctions.optimize_forward_curve_spline(
        df = df,
        knots = knots,
        error_func = ModellingFunctions.fit_cubic_forward_curve_error,
        bounds=(0.0001, 0.1),
        method='L-BFGS-B',
        options={"maxiter": 200}
    )
    
    # Generate dense output
    max_fit_term = df["term_yr"].max()
    output_terms = np.arange(0, max_fit_term + 1/12, 1/12)
    fwd_rates = spline(output_terms)

    df_cubic = pd.DataFrame({
        "term_yr": output_terms,
        "fwd_rate_cubic": fwd_rates
    })

    df_all = pd.merge_asof(df_cubic, df_bootstrapped, on="term_yr", direction="nearest")
    df_all = df_all.sort_values(by="term_yr").bfill()

    # Merge on market observed spot rates
    df_all = df_all.merge(df[["term_yr", "spot_rate_pa"]], on="term_yr", how="left")
    
    return df_all


def ask_chatgpt_for_knot_tweaks(fig, previous_response_text):
    try:
        # Convert fig to base64 image
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        prompt = (
            "You previously suggested these knots for fitting a cubic spline to a forward rate curve:\n\n"
            f"{previous_response_text}\n\n"
            "Here is a plot showing the resulting fit using those knots.\n\n"
            "Based on the shape of the curve and any visible overfitting or underfitting, "
            "would you make any adjustments to the knots? If so, return a new list of 4 knots "
            "with brief justification for the change.\n\n"
            "Please return just:\n"
            "- An initial comment on whether there is visible overfitting or underfitting, where this is and what is likely causing it\n"
            "- A new list of 4 knot positions in years\n"
            "- A one-line justification for any changes to knot positions"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial modeling assistant for yield curve fitting."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            max_tokens=700
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"ChatGPT knot refinement request failed: {e}")
        return None


# Bridging forward curve
def bootstrap_forward_curve(df_curve, long_term_rate, max_slope, extension_freq, min_extension_years, curve_max_extension):
    df_extended = ModellingFunctions.bridge_forward_curve_to_longterm(
        df_curve = df_curve,
        term_col = "term_yr",
        fwd_col = "fwd_rate_cubic",
        long_term_rate = long_term_rate,
        max_slope = max_slope,
        extension_freq = extension_freq,
        min_extension_years = min_extension_years
    )

    # Expand to 100 years at 1/12 intervals
    full_terms = np.arange(0, curve_max_extension, 1/12)

    # Forward fill the forward rates
    full_fwd_rates = np.interp(full_terms, df_extended["term_yr"], df_extended["fwd_rate_extended"])

    # Final output
    df_extended = pd.DataFrame({
        "term_yr": full_terms,
        "fwd_rate_extended": full_fwd_rates,
    })
    
    return df_extended


# Main Page Content

st.set_page_config(layout="wide")
st.title("🧮 Create Risk-free Discount Curve")

st.subheader("Bootstrapping of zero-coupon forward rates")
st.markdown("""
The one-month forward rate is determined from the Overnight Cash Rate (OCR).

Nominal Government bonds are decomposed into maturity and individual coupon payments to produce a set of equivalent zero-coupon nominal bonds maturing on the 15th of the month

A forward rate is determined for the shortest nominal Government bond, for the period up until the first nominal bond matures. For the period between the first nominal bond and the nominal second bond a forward rate is determined so that the second nominal bond market value is equalled using the previous forward rates as well. This process is repeated to solve for each successive forward rate until all nominal bonds have been valued.            
""")

pdf_yields, pdf_amounts, as_of_date = load_data_from_session()

# Merge and clean
merged_pdf = pdf_yields.merge(pdf_amounts, on='series_id', how='left')
merged_pdf['term_yr'] = merged_pdf['term_mth_whole'].replace(0, 1) / 12

if st.button("🚀 Run Forward Bootstrapping"):
    try:
        df_bootstrapped = bootstrap_forward_curve(merged_pdf)
        st.success("✅ Bootstrapping complete!")

        st.subheader("📈 Bootstrapped Curve Output")
        #fig = plot_forward_curve(df_bootstrapped)
        fig = plot_forward_curve(df_bootstrapped,
                x_col = "term_yr",
                line1_col = "fwd_rate_bootstrapped",
                line1_name = "Forward rate (bootstrapped)",
                scatter_col = "yield_decimal",
                scatter_name = "Market Yields",
                show_step_lines = True)
        st.session_state["bootstrapped_fig"] = fig
        st.session_state["bootstrapped_data"] = df_bootstrapped 
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_bootstrapped)
    except Exception as e:
        st.error(f"Bootstrapping failed: {e}")

fig = st.session_state.get("bootstrapped_fig", None)
data = st.session_state.get("bootstrapped_data", None)

if fig and st.button("🧠 Ask ChatGPT for Knot Placement Advice"):
    st.plotly_chart(fig, use_container_width=True)
    st.info("Sending chart to ChatGPT...")
    advice = ask_chatgpt_for_knot_advice(fig, data)
    if advice:
        st.success("Response received!")
        st.markdown("### 💡 Knot Placement Advice for Cubic Spline")
        st.write(advice)
        
        st.session_state["previous_knot_advice"] = advice

        knot_list = extract_knot_list(advice)

        if knot_list and len(knot_list) == 4:
            st.session_state["spline_knots"] = knot_list
            st.success(f"Knots extracted: {knot_list}")
        else:
            st.error("Could not extract 4 valid knot values from the response.")

st.subheader("📐 Cubic Spline Curve Fitting")
st.markdown("""
The process is to fit a curve of forward rates to the zero-coupon portfolio of available bonds. The parameters of the fitted curve are determined by solving to minimize the least squares differences of the resulting fitted spot rates with the actual market spot rates. Two-, three- and six-month Treasury bill rates are used in addition to nominal Government bonds.

Market yields are weighted by the lesser of the amount available in the market, which excludes the amounts held by the Reserve Bank of New Zealand (RBNZ) and the Earthquake Commission (which is not usually traded) and $4 billion. This means that implied forward rates automatically give less weight to those bonds which represent a smaller proportion of the tradeable market.

The curve fitted is a cubic spline on the forward rates with 4 knots. This is fairly standard methodology with enough flexibility to fit most yield curves. There is some judgment involved in selecting the position of the knots, but this also gives a little flexibility to cope with any anomalies that may be present in the yield curve without changing the fundamental principles.            
""")
if st.button("🚀 Run Cubic Spline Fitting"):
    try:
        knots = st.session_state.get("spline_knots", None)
        df_bootstrapped = st.session_state.get("bootstrapped_data", None)
        df_all = fit_cubic_spline_forward(merged_pdf, knots, df_bootstrapped)
        st.success(f"✅ Cubic Spline fitting complete! Knots used: {knots}")
        fig = plot_forward_curve(df_all,
                    x_col = "term_yr",
                    line1_col = "fwd_rate_bootstrapped",
                    line1_name = "Forward rate (bootstrapped)",
                    line2_col = "fwd_rate_cubic",
                    line2_name = "Forward rate (cubic spline)",
                    show_step_lines = True)
        for knot in knots:
            fig.add_vline(x=knot, line_dash="dash", line_color="gray")
        st.session_state["fitted_figure"] = fig
        st.session_state["fitted_data"] = df_all
        st.subheader("📈 Fitted Curve Output")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("Users can iterate between running the cubic spline fitting and asking ChatGPT to iterate until you are happy with the selection. Alternatively, enter your own knot placements and re-run.")

    except Exception as e:
        st.error(f"Curve fitting failed: {e}")
    
col1, col2 = st.columns(2)

# 📤 Column 1: GPT-based knot refinement
with col1:
    st.markdown("#### 🔁 Refine Knots with ChatGPT")

    if st.button("🔁 Ask ChatGPT to Refine Knots"):
        if "previous_knot_advice" not in st.session_state:
            st.warning("You must run the first knot suggestion before refining.")
        else:
            fig = st.session_state.get("fitted_figure")

            if fig is None:
                st.error("No fitted curve found. Please run the spline fitting first.")
            else:
                new_advice = ask_chatgpt_for_knot_tweaks(fig, st.session_state["previous_knot_advice"])

                if new_advice:
                    st.markdown("### 🧠 GPT's Suggested Knot Adjustments")
                    st.write(new_advice)

                    knot_list = extract_knot_list(new_advice)
                    if knot_list and len(knot_list) == 4:
                        st.session_state["spline_knots"] = knot_list
                        st.success(f"Updated knots: {knot_list}")
                    else:
                        st.error("Could not extract a valid list of 4 knots.")

# ✍️ Column 2: User-defined knots
with col2:
    st.markdown("#### ✍️ Enter Your Own Knots")
    user_input = st.text_input("Enter 4 knot positions (comma-separated, e.g., 1, 4.5, 12, 25)")

    if st.button("📌 Use Custom Knots"):
        try:
            user_knots = [float(x.strip()) for x in user_input.split(",")]
            if len(user_knots) == 4:
                st.session_state["spline_knots"] = user_knots
                st.success(f"✅ Custom knots set: {user_knots}")
            else:
                st.error("❌ Please enter exactly 4 numbers.")
        except ValueError:
            st.error("❌ Invalid input. Make sure to enter numbers separated by commas.")


st.subheader("📐 Apply Bridging to Long-term Assumption")
st.markdown("""
Bridging is required from the last observable market data point, out to a long-term assumption. The methodology applies linear interpolation over a defined period from the maturity date of the last nominal Government bond, subject to a defined maximum slope.
""")

# Main layout for bridging
left_col, right_col = st.columns([1, 3])

with left_col:

    with st.form("bridging_form"):
        
        df_curve = st.session_state.get("fitted_data")

        long_term_rate = st.number_input("Long-term forward rate", value=0.048, step=0.0001, format="%.6f")
        bridging_max_slope = st.number_input("Max slope of extension", value=0.0005, step=0.0001, format="%.6f")
        bridging_min_extension_years = st.number_input("Min extension (years)", value=10.0, step=0.5)
        bridging_extension_freq = st.number_input("Extension frequency (years)", value=0.25, step=0.05)
        curve_max_extension = st.number_input("Max extension (years)", value=100, step=1)

        run_bridging = st.form_submit_button("🌐 Run Bridging")

with right_col:
    if run_bridging:
        try:
            result_df = bootstrap_forward_curve(df_curve, long_term_rate, bridging_max_slope, bridging_extension_freq, bridging_min_extension_years, curve_max_extension)

            st.success("✅ Bridging completed!")
            fig = plot_forward_curve(result_df,
                        x_col = "term_yr",
                        line1_col = "fwd_rate_extended",
                        line1_name = "Forward rate (final, fitted)",
                        show_step_lines = True)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Bridging failed: {e}")

if run_bridging:
    st.markdown("Final forward curve output available to download:")
    st.dataframe(result_df)








