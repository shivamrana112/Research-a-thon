# app.py
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------
# Page config & Apple-style CSS
# -----------------------
st.set_page_config(page_title="Battery Life Dashboard — Apple Style", layout="wide")

st.markdown("""
<style>
/* Base */
:root {
  --bg: #f7f8fa;
  --card: #ffffff;
  --muted: #6b7280;
  --accent: #0b69ff;
  --radius: 12px;
  --shadow: 0 6px 18px rgba(20,20,20,0.06);
  --small: 13px;
}

/* Page background */
.reportview-container, .main {
  background: var(--bg);
}

/* Card */
.card {
  background: var(--card);
  border-radius: var(--radius);
  padding: 18px;
  box-shadow: var(--shadow);
  margin-bottom: 18px;
}

/* Headings */
h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
h1 { font-size: 22px; margin-bottom: 0.2rem; }
h2 { font-size: 16px; color: #111827; }
h3 { font-size: 14px; color: #374151; }

/* Sidebar */
[data-testid="stSidebar"] .css-1d391kg {
    background-color: transparent;
}
.sidebar .stButton>button {
    border-radius: 10px;
}

/* Reduce default padding for neatness */
.block-container {
    padding-top: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Small muted text */
.muted { color: var(--muted); font-size: var(--small); }

/* Compact table dispmissing required columns — provide a compatible CSV.lay */
.streamlit-expanderHeader, .st-bx {
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Constants & dataset path
# -----------------------
DATA_FILE = "battery_dataset.csv"

# -----------------------
# Generate dataset (once) if missing
# -----------------------
def generate_sample_dataset(path=DATA_FILE, N=1500, seed=42):
    np.random.seed(seed)
    data = {
        "Cycle_Index": np.arange(1, N + 1),
        "Discharge_Time(s)": np.random.uniform(1200, 5000, N),
        "Decrement_3.6-3.4V(s)": np.random.uniform(40, 200, N),
        "Max_Voltage_Discharge(V)": np.random.uniform(3.85, 4.25, N),
        "Min_Voltage_Charger(V)": np.random.uniform(2.5, 3.25, N),
    }
    data["RUL"] = (1500 - data["Cycle_Index"]) + np.random.normal(0, 40, N)
    data["RUL"] = np.clip(data["RUL"], 0, None)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df

@st.cache_data(ttl=24*3600)
def load_or_create_dataset(path=DATA_FILE):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return generate_sample_dataset(path)

# load dataset (cached)
df = load_or_create_dataset()

# -----------------------
# Material database (keep all materials)
# -----------------------
materials = {
    "Lithium-ion (NMC/LFP)": {
        "Energy Density": 8, "Cycle Life": 8, "Cost": 5, "Environmental Impact": 4, "Safety": 6,
        "Notes": "High energy density; recycling improving; mining impacts for Li/Co."
    },
    "Sodium-ion": {
        "Energy Density": 6, "Cycle Life": 6, "Cost": 8, "Environmental Impact": 8, "Safety": 8,
        "Notes": "Abundant materials; lower density; promising for grid storage."
    },
    "Solid-State": {
        "Energy Density": 10, "Cycle Life": 9, "Cost": 4, "Environmental Impact": 6, "Safety": 10,
        "Notes": "Great safety and density potential; manufacturing challenges."
    },
    "Silicon Anode": {
        "Energy Density": 10, "Cycle Life": 6, "Cost": 5, "Environmental Impact": 5, "Safety": 6,
        "Notes": "Boosts energy density when used with Li-ion; expansion issues under research."
    },
    "Lithium–Sulfur": {
        "Energy Density": 10, "Cycle Life": 4, "Cost": 7, "Environmental Impact": 7, "Safety": 5,
        "Notes": "Very high theoretical energy; cycle life and polysulfide shuttle are challenges."
    },
    "Graphene Enhanced": {
        "Energy Density": 9, "Cycle Life": 9, "Cost": 4, "Environmental Impact": 6, "Safety": 8,
        "Notes": "Additive to improve conductivity and cycle life; manufacturing cost factor."
    },
    "Aluminum-ion": {
        "Energy Density": 6, "Cycle Life": 9, "Cost": 9, "Environmental Impact": 8, "Safety": 9,
        "Notes": "Fast charging and long life promising for some grid applications."
    },
    "Zinc-air": {
        "Energy Density": 7, "Cycle Life": 5, "Cost": 9, "Environmental Impact": 9, "Safety": 9,
        "Notes": "Good for low-cost applications and recyclability; still research-level for rechargeables."
    }
}

# -----------------------
# Cached model training
# -----------------------
@st.cache_resource
def train_model_and_scaler(dataframe):
    required_cols = ["Cycle_Index", "Discharge_Time(s)", "Decrement_3.6-3.4V(s)",
                     "Max_Voltage_Discharge(V)", "Min_Voltage_Charger(V)", "RUL"]
    if not all(c in dataframe.columns for c in required_cols):
        raise ValueError("Dataset missing required columns — provide a compatible CSV.")
    X = dataframe[required_cols[:-1]]
    y = dataframe["RUL"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

try:
    model, scaler = train_model_and_scaler(df)
except Exception as e:
    st.error(f"Model training error: {e}")
    st.stop()

# -----------------------
# Sidebar: navigation (Apple-like icons)
# -----------------------
st.sidebar.markdown("<div style='font-size:16px;font-weight:600'>🔋 Battery Dashboard</div>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Dataset & Model", "Simulation & Animation", "Materials & Environment"], index=0)

# small helper to render card
def card(title, body_fn):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### {title}")
    body_fn()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Page: Dataset & Model
# -----------------------
if page == "Dataset & Model":
    card("Dataset preview & options", lambda: (
        st.write("**Loaded dataset (first 8 rows)**"),
        st.dataframe(df.head(8).round(3)),
        st.markdown("<span class='muted'>Tip: upload a compatible CSV to the same folder named 'Battery_RUL.csv' to replace the dataset permanently.</span>", unsafe_allow_html=True)
    ))

    def model_info():
        st.write("**Model:** RandomForestRegressor (cached).")
        st.write("- Trained on dataset features: Cycle_Index, Discharge_Time(s), Decrement_3.6-3.4V(s), Max_Voltage_Discharge(V), Min_Voltage_Charger(V).")
        st.write("- Target: RUL (Remaining Useful Life)")
        st.write("")
        st.download_button("Download sample dataset CSV", data=df.to_csv(index=False), file_name="Battery_RUL_sample.csv")

    card("Model & info", model_info)

# -----------------------
# Page: Simulation & Animation
# -----------------------
if page == "Simulation & Animation":
    def sim_controls():
        st.write("Choose starting row, charging policy and material. Animations and plots are interactive (Plotly).")

    card("Simulation controls", sim_controls)

    # controls
    col1, col2 = st.columns([1, 1])
    with col1:
        start_idx = st.number_input("Starting row index (0-based)", min_value=0, max_value=len(df)-1, value=0, step=1)
        policy = st.selectbox("Charging policy", ["Normal", "Fast", "Slow"])
        material_choice = st.selectbox("Battery material", list(materials.keys()), index=0)
    with col2:
        run_button = st.button("Run Simulation")
        st.write("Material notes:")
        st.write(materials[material_choice]["Notes"])

    # deterministic input
    init_row = df.loc[start_idx, ["Cycle_Index", "Discharge_Time(s)", "Decrement_3.6-3.4V(s)",
                                  "Max_Voltage_Discharge(V)", "Min_Voltage_Charger(V)"]].to_numpy()
    init_scaled = scaler.transform([init_row])[0]

    policy_factors = {"Fast": 0.85, "Normal": 1.0, "Slow": 1.12}
    material_eff = materials[material_choice]["Energy Density"] / 8.0

    if run_button:
        steps = 180
        rul_series = []
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Predicted RUL'))
        fig.update_layout(title=f"RUL under {policy} charging — Material: {material_choice}",
                          xaxis_title="Cycle step", yaxis_title="Predicted RUL (cycles)",
                          height=420, template="simple_white",
                          margin=dict(t=50, b=40, l=40, r=20))

        placeholder = st.empty()
        for t in range(steps):
            scaled_state = init_scaled.copy()
            scaled_state[0] += t * 0.01
            scaled_state[1] *= (1 - (0.001 * policy_factors[policy]))
            scaled_state[2] *= (1 + (0.0003 * policy_factors[policy]))
           # ---- MATERIAL PROPERTIES ----
            material_data = {
    "Lithium-ion (NMC/LFP)": {"conductivity": 0.9, "degradation_rate": 0.35, "thermal_stability": 0.7},
    "Sodium-ion": {"conductivity": 0.55, "degradation_rate": 0.50, "thermal_stability": 0.45},
    "Solid-State": {"conductivity": 1.25, "degradation_rate": 0.20, "thermal_stability": 1.45},
    "Silicon Anode": {"conductivity": 1.10, "degradation_rate": 0.55, "thermal_stability": 0.55},
    "Lithium–Sulfur": {"conductivity": 1.30, "degradation_rate": 0.65, "thermal_stability": 0.50},
    "Graphene Enhanced": {"conductivity": 1.60, "degradation_rate": 0.15, "thermal_stability": 1.70},
    "Aluminum-ion": {"conductivity": 0.70, "degradation_rate": 0.30, "thermal_stability": 1.20},
    "Zinc-air": {"conductivity": 0.45, "degradation_rate": 0.55, "thermal_stability": 0.50}
}

            mp = material_data[material_choice]

# ---- MATERIAL IMPACT (BIG & VISIBLE) ----
            material_factor = (
                1
                + mp["conductivity"] * 0.22
                - mp["degradation_rate"] * 0.30
                + mp["thermal_stability"] * 0.18
            )

# ---- CHARGING POLICY IMPACT ----
            policy_factor = {
                "Fast": 0.70,    # Fast charging reduces RUL strongly
                "Normal": 1.00,
                "Slow": 1.18     # Slow charging increases RUL
            }[policy]

# ---- BASE PREDICTION ----
            base_pred = model.predict([scaled_state])[0]

# ---- FINAL ADJUSTED RUL ----
            pred = base_pred * material_factor * policy_factor

            rul_series.append(pred)
            fig.data[0].x = list(range(len(rul_series)))
            fig.data[0].y = rul_series
            placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.03)

        # After simulation, export results button
        res_df = pd.DataFrame({"cycle_step": np.arange(len(rul_series)), "predicted_RUL": rul_series})
        st.download_button("Download simulation results (CSV)", data=res_df.to_csv(index=False), file_name="simulation_results.csv")
    else:
        # show a compact static preview for the chosen policy/material
        base_pred = model.predict([init_scaled])[0] * policy_factors[policy] * material_eff
        st.metric(label="Sample predicted RUL (single-step)", value=f"{base_pred:.1f} cycles")
        # small static curve preview
        preview_y = np.linspace(base_pred, base_pred - 90, 100)
        preview_fig = px.line(x=list(range(100)), y=preview_y, labels={"x": "Cycle step", "y": "Predicted RUL"},
                              title="Preview degradation (static)", height=300)
        preview_fig.update_layout(template="simple_white", margin=dict(t=40, b=30))
        st.plotly_chart(preview_fig, use_container_width=True)

    # 3D rotating surface (precomputed)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.subheader("3D Degradation Surface")
    cycles = np.linspace(1, 300, 80)
    temps = np.linspace(20, 60, 40)
    C, T = np.meshgrid(cycles, temps)
    Z = 900 - (C ** 0.7) - (T * 1.1)

    frames = []
    for angle in np.linspace(0, 360, 36):
        frames.append(go.Frame(data=[go.Surface(z=Z, x=C, y=T, colorscale='Viridis')], layout=go.Layout(scene_camera=dict(eye=dict(x=1.7*np.cos(np.radians(angle)),
                                                                                                                      y=1.7*np.sin(np.radians(angle)),
                                                                                                                      z=0.8)))))

    fig3d = go.Figure(data=[go.Surface(z=Z, x=C, y=T, colorscale='Viridis')], frames=frames)
    fig3d.update_layout(scene=dict(xaxis_title='Cycles', yaxis_title='Temperature (°C)', zaxis_title='RUL'),
                        height=420, margin=dict(l=0, r=0, t=30, b=0))
    fig3d.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                         y=1.05, x=1.15, xanchor='right', yanchor='top',
                                         buttons=[dict(label='Play', method='animate',
                                                       args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True, mode='immediate')])])])
    st.plotly_chart(fig3d, use_container_width=True)

# -----------------------
# Page: Materials & Environment
# -----------------------
if page == "Materials & Environment":
    st.subheader("Materials & Environmental Effects")
    left, right = st.columns([1, 1])

    with left:
        selected = st.selectbox("Select material", list(materials.keys()))
        info = materials[selected]
        st.markdown(f"**{selected}**")
        st.write(info["Notes"])
        st.markdown("**Advantages**")
        st.write("- " + "\n- ".join([a for a in ["High energy density (where applicable)", "Mature chemistries", "Safety improvements"] if a]))
        st.markdown("**Considerations**")
        st.write("- Mining footprint, recycling needs, cost and manufacturability vary by chemistry.")

    with right:
        # Radar chart for selected material
        labels = ['Energy Density', 'Cycle Life', 'Cost', 'Environmental Impact', 'Safety']
        values = [info[l] for l in labels]
        values += values[:1]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=values, theta=labels+labels[:1], fill='toself', name=selected))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=False, height=380)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Environmental impact comparison of all materials
    st.markdown("---")
    st.write("Environmental impact comparison (higher = better eco-profile in this simplified scale)")
    env_df = pd.DataFrame([{"material": k, "env": v["Environmental Impact"]} for k, v in materials.items()])
    bar = px.bar(env_df, x='material', y='env', labels={'env': 'Environmental Score'}, height=320)
    bar.update_layout(xaxis_tickangle=-35, template="simple_white")
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("**Quick guidance:**")
    st.markdown("""
- **Lithium-ion:** Best general performance; push recycling and responsible sourcing.  
- **Solid-state:** Safety + energy promise — manufacturing & cost are current barriers.  
- **Sodium-ion:** Attractive for low-cost grid storage; lower energy density but better resource profile.  
- **Graphene / Silicon-anode / Lithium–Sulfur / Aluminum-ion / Zinc-air:** Specialized trade-offs — good for certain niches; consider lifecycle & recycling.
""")

# -----------------------
# Footer
# -----------------------
# st.markdown("<div style='text-align:center; padding:14px 0; color:#6b7280;'>Made with ❤️ — Apple-style Streamlit dashboard • Data & model cached for stability</div>", unsafe_allow_html=True)
