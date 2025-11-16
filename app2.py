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
# Page config & Enterprise theme CSS
# -----------------------
st.set_page_config(page_title="Battery Dashboard — Enterprise", layout="wide")

st.markdown("""
<style>
/* Enterprise theme */
:root{
  --bg:#f4f6f9;
  --card:#ffffff;
  --muted:#637381;
  --accent:#0b5fff;
  --accent-2:#0077b6;
  --radius:10px;
  --shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
  --small:13px;
  --heading: 'Segoe UI', Roboto, "Helvetica Neue", Arial;
}

/* page */
.reportview-container .main {
  background: var(--bg);
  color: #0b1724;
  font-family: var(--heading);
}

/* card */
.card {
  background: var(--card);
  border-radius: var(--radius);
  padding: 18px;
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}

/* header */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}
.app-title { font-size:20px; font-weight:700; color:#0b1724; }
.app-sub { color:var(--muted); font-size:13px; }

/* sidebar styling */
[data-testid="stSidebar"] .css-1d391kg { background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%); border-radius: 10px; padding: 12px; }
.stButton>button { border-radius: 8px; }

/* small text */
.muted { color: var(--muted); font-size: var(--small); }

/* compact table */
.streamlit-expanderHeader, .st-bx { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Constants & dataset path
# -----------------------
DATA_FILE = "Battery_RUL.csv"

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
# Material database (kept and extended)
# -----------------------
materials = {
    "Lithium-ion (NMC/LFP)": {"Energy Density": 8, "Cycle Life": 8, "Cost": 5, "Environmental Impact": 4, "Safety": 6, "Notes": "High energy density; recycling improving; mining impacts for Li/Co."},
    "Sodium-ion": {"Energy Density": 6, "Cycle Life": 6, "Cost": 8, "Environmental Impact": 8, "Safety": 8, "Notes": "Abundant materials; lower density; promising for grid storage."},
    "Solid-State": {"Energy Density": 10, "Cycle Life": 9, "Cost": 4, "Environmental Impact": 6, "Safety": 10, "Notes": "Great safety and density potential; manufacturing challenges."},
    "Silicon Anode": {"Energy Density": 10, "Cycle Life": 6, "Cost": 5, "Environmental Impact": 5, "Safety": 6, "Notes": "Boosts energy density when used with Li-ion; expansion issues under research."},
    "Lithium–Sulfur": {"Energy Density": 10, "Cycle Life": 4, "Cost": 7, "Environmental Impact": 7, "Safety": 5, "Notes": "Very high theoretical energy; cycle life and polysulfide shuttle are challenges."},
    "Graphene Enhanced": {"Energy Density": 9, "Cycle Life": 9, "Cost": 4, "Environmental Impact": 6, "Safety": 8, "Notes": "Additive to improve conductivity and cycle life; manufacturing cost factor."},
    "Aluminum-ion": {"Energy Density": 6, "Cycle Life": 9, "Cost": 9, "Environmental Impact": 8, "Safety": 9, "Notes": "Fast charging and long life promising for some grid applications."},
    "Zinc-air": {"Energy Density": 7, "Cycle Life": 5, "Cost": 9, "Environmental Impact": 9, "Safety": 9, "Notes": "Good for low-cost applications and recyclability; still research-level for rechargeables."}
}

# -----------------------
# Cached model training
# -----------------------
@st.cache_resource
def train_model_and_scaler(dataframe):
    req = ["Cycle_Index", "Discharge_Time(s)", "Decrement_3.6-3.4V(s)", "Max_Voltage_Discharge(V)", "Min_Voltage_Charger(V)", "RUL"]
    if not all(c in dataframe.columns for c in req):
        raise ValueError("Dataset missing required columns — provide a compatible CSV.")
    X = dataframe[req[:-1]]
    y = dataframe["RUL"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=160, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

try:
    model, scaler = train_model_and_scaler(df)
except Exception as e:
    st.error(f"Model training error: {e}")
    st.stop()

# -----------------------
# Sidebar: navigation
# -----------------------
st.sidebar.markdown("<div style='font-size:16px; font-weight:700; color:#0b1724;'>⚙ Battery Analytics</div>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Overview", "Dataset & Model", "Simulation", "Materials"], index=0)

# small helper to render card
def card(title, body_fn):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### {title}")
    body_fn()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# PAGE: Overview
# -----------------------
if page == "Overview":
    def overview_body():
        st.write("**Enterprise Battery Analytics Dashboard**")
        st.write("Use the left menu to explore dataset, simulation and material impacts. The model and data are cached to keep results stable across navigation.")
        # KPIs row
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset rows", f"{len(df):,}")
        c2.metric("Features", "5")
        c3.metric("Model", "RandomForest")
        # show small correlation heatmap
        corr = df.corr()
        fig = px.imshow(corr, color_continuous_scale="Blues", title="Feature Correlation", height=300)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    card("Overview", overview_body)

# -----------------------
# PAGE: Dataset & Model
# -----------------------
if page == "Dataset & Model":
    def ds_body():
        st.write("**Dataset preview (first 10 rows)**")
        st.dataframe(df.head(10).round(3))
        st.markdown("**Model info**")
        st.write("- RandomForestRegressor trained on common battery features.")
        st.write("- Target: Remaining Useful Life (RUL).")
        st.download_button("Download sample dataset", data=df.to_csv(index=False), file_name="Battery_RUL_sample.csv")
        st.markdown("Upload compatible CSV named 'Battery_RUL.csv' in the working folder to replace dataset permanently.")
    card("Dataset & Model", ds_body)

# -----------------------
# PAGE: Simulation
# -----------------------
if page == "Simulation":
    def sim_ctrl():
        st.write("Select starting row, charging policy and material. Click RUN to simulate RUL over cycles.")
    card("Controls", sim_ctrl)

    left, right = st.columns([1, 1])
    with left:
        start_idx = st.number_input("Start row index (0-based)", min_value=0, max_value=len(df)-1, value=0, step=1)
        policy = st.selectbox("Charging policy", ["Normal", "Fast", "Slow"])
        materials_select = st.selectbox("Battery material", list(materials.keys()), index=0)
        run_sim = st.button("RUN SIMULATION", key="run_sim")
    with right:
        st.write("Material notes")
        st.write(materials[materials_select]["Notes"])

    # deterministic input and scaling
    init_row = df.loc[start_idx, ["Cycle_Index", "Discharge_Time(s)", "Decrement_3.6-3.4V(s)", "Max_Voltage_Discharge(V)", "Min_Voltage_Charger(V)"]].to_numpy()
    init_scaled = scaler.transform([init_row])[0]

    # stronger material model to make effects visible
    material_properties = {
        "Lithium-ion (NMC/LFP)": {"cond":0.9, "deg":0.35, "therm":0.7},
        "Sodium-ion": {"cond":0.55, "deg":0.50, "therm":0.45},
        "Solid-State": {"cond":1.25, "deg":0.20, "therm":1.45},
        "Silicon Anode": {"cond":1.10, "deg":0.55, "therm":0.55},
        "Lithium–Sulfur": {"cond":1.30, "deg":0.65, "therm":0.50},
        "Graphene Enhanced": {"cond":1.60, "deg":0.15, "therm":1.70},
        "Aluminum-ion": {"cond":0.70, "deg":0.30, "therm":1.20},
        "Zinc-air": {"cond":0.45, "deg":0.55, "therm":0.50}
    }

    policy_map = {"Fast":0.70, "Normal":1.00, "Slow":1.18}

    # Run & show animation / chart
    if run_sim:
        steps = 200
        rul_series = []
        fig = go.Figure(layout=dict(template="plotly_white", height=450))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color=st.session_state.get("accent_color", "#0b5fff")), name='Predicted RUL'))
        fig.update_layout(title=f"Simulated RUL — Material: {materials_select} | Policy: {policy}", xaxis_title="Cycle step", yaxis_title="Predicted RUL (cycles)", margin=dict(t=60,b=40,l=40,r=20))
        placeholder = st.empty()

        mp = material_properties[materials_select]
        # compute material factor (scaled to be impactful)
        material_factor = (1 + mp["cond"]*0.25 - mp["deg"]*0.32 + mp["therm"]*0.2)
        policy_factor = policy_map[policy]

        for t in range(steps):
            s_state = init_scaled.copy()
            s_state[0] += t * 0.015
            s_state[1] *= (1 - (0.0012 * (1/policy_factor)))  # make fast degrade faster
            s_state[2] *= (1 + (0.0004 * (1/policy_factor)))

            base_pred = model.predict([s_state])[0]
            pred = base_pred * material_factor * policy_factor
            rul_series.append(pred)

            fig.data[0].x = list(range(len(rul_series)))
            fig.data[0].y = rul_series
            placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.02)

        # download
        res_df = pd.DataFrame({"cycle_step": np.arange(len(rul_series)), "predicted_RUL": rul_series})
        st.download_button("Download simulation CSV", data=res_df.to_csv(index=False), file_name="simulation_results.csv")
    else:
        # static preview
        mp = material_properties[materials_select]
        material_factor = (1 + mp["cond"]*0.25 - mp["deg"]*0.32 + mp["therm"]*0.2)
        policy_factor = policy_map[policy]
        base_pred = model.predict([init_scaled])[0]
        sample_pred = base_pred * material_factor * policy_factor
        st.metric(label="Sample predicted RUL (single-step)", value=f"{sample_pred:.1f} cycles")

        preview = np.linspace(sample_pred, sample_pred - 120*policy_factor, 140)
        preview_fig = px.line(x=list(range(len(preview))), y=preview, labels={"x":"Cycle step","y":"Predicted RUL"}, title="Static Preview")
        preview_fig.update_traces(line=dict(color="#0b5fff"))
        preview_fig.update_layout(height=300, template="plotly_white")
        st.plotly_chart(preview_fig, use_container_width=True)

    # 3D interactive surface
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.subheader("3D Degradation Surface")
    cycles = np.linspace(1, 300, 80)
    temps = np.linspace(20, 60, 40)
    C, T = np.meshgrid(cycles, temps)
    Z = 850 - (C ** 0.75) - (T * 1.08)

    fig3d = go.Figure(data=[go.Surface(z=Z, x=C, y=T, colorscale='Blues')])
    fig3d.update_layout(scene=dict(xaxis_title='Cycles', yaxis_title='Temp (°C)', zaxis_title='RUL'), height=420, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig3d, use_container_width=True)

# -----------------------
# PAGE: Materials
# -----------------------
if page == "Materials":
    def materials_body():
        st.write("Materials: properties, pros/cons and environmental notes.")
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
            # radar chart
            labels = ['Energy Density', 'Cycle Life', 'Cost', 'Environmental Impact', 'Safety']
            values = [info[l] for l in labels]
            values += values[:1]
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(r=values, theta=labels+labels[:1], fill='toself', name=selected))
            radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=False, height=360, template="plotly_white")
            st.plotly_chart(radar, use_container_width=True)
            # env bar
            env_df = pd.DataFrame([{"material": k, "env": v["Environmental Impact"]} for k,v in materials.items()])
            bar = px.bar(env_df, x='material', y='env', labels={'env':'Environmental Score'}, title="Environmental Impact (simplified)", height=250)
            bar.update_layout(xaxis_tickangle=-30, template="plotly_white")
            st.plotly_chart(bar, use_container_width=True)
    card("Materials & Environment", materials_body)

# -----------------------
# Footer
# -----------------------
st.markdown("<div style='text-align:center; padding:14px 0; color:#637381;'>© Battery Analytics — Enterprise View • Data/model cached for stability</div>", unsafe_allow_html=True)
