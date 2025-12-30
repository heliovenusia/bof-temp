import json, time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="BOF Tap Temperature Prediction", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("artifacts/feature_schema.json","r") as f:
        schema = json.load(f)
    model = tf.keras.models.load_model("artifacts/bof_temp_model.keras")
    features = schema["features_ordered"]
    units = schema.get("units", {})
    ranges = schema.get("ranges", {})
    mu = np.array(schema["standardization_params"]["mean"], dtype=np.float32)
    sigma = np.array(schema["standardization_params"]["std"], dtype=np.float32)
    return model, schema, features, units, ranges, mu, sigma

model, schema, FEATURES, UNITS, RANGES, MU, SIGMA = load_artifacts()

def build_default_input():
    d = {}
    for f in FEATURES:
        r = RANGES.get(f, None)
        if r and np.isfinite(r.get("min", np.nan)) and np.isfinite(r.get("max", np.nan)):
            d[f] = float((r["min"] + r["max"]) / 2.0)
        else:
            d[f] = 0.0
    return d

def preprocess_payload(payload):
    x = np.array([payload[f] for f in FEATURES], dtype=np.float32)
    idx_hm = FEATURES.index("hot_metal_weight")
    x[idx_hm] = x[idx_hm] * 1000.0
    x = (x - MU) / SIGMA
    return x.reshape(1, -1)

def predict_temp(payload):
    x = preprocess_payload(payload)
    y = float(model.predict(x, verbose=0).reshape(-1)[0])
    return y

def sensitivity_contrib(payload, frac=0.01):
    base = predict_temp(payload)
    contrib = []
    for f in FEATURES:
        p2 = dict(payload)
        v = float(p2[f])
        delta = abs(v) * frac
        if delta < 1e-6:
            delta = frac
        p2[f] = v + delta
        y2 = predict_temp(p2)
        contrib.append((f, y2 - base))
    dfc = pd.DataFrame(contrib, columns=["feature","delta_temp_C"])
    dfc["abs_delta"] = dfc["delta_temp_C"].abs()
    dfc = dfc.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"]).reset_index(drop=True)
    return base, dfc

st.title("BOF Tap Temperature Prediction")

tab1, tab2, tab3 = st.tabs(["Prediction Console", "Insights (Parameter Contribution)", "API View"])

if "payload" not in st.session_state:
    st.session_state.payload = build_default_input()

with tab1:
    left, right = st.columns([0.68, 0.32], gap="large")
    with left:
        st.subheader("Process Inputs (Manual for demo)")
        payload = dict(st.session_state.payload)

        def field(label, key):
            r = RANGES.get(key, {})
            mn = r.get("min", None)
            mx = r.get("max", None)
            u = UNITS.get(key, "")
            step = None
            if mx is not None and mn is not None and np.isfinite(mx) and np.isfinite(mn):
                span = float(mx - mn)
                step = span / 200.0 if span > 0 else 0.1
            v = st.number_input(f"{label} ({u})", value=float(payload[key]), step=step if step and step > 0 else 0.1, format="%.6f")
            hint = ""
            if mn is not None and mx is not None and np.isfinite(mn) and np.isfinite(mx):
                hint = f"Observed range: {mn:.4g} – {mx:.4g} {u}"
            if hint:
                st.caption(hint)
            payload[key] = float(v)

        with st.expander("Hot Metal & Chemistry", expanded=True):
            field("Silicon", "silicon")
            field("Manganese", "Mn")
            field("Carbon", "C")
            field("Sulphur", "S")
            field("Phosphorus", "P")

        with st.expander("Weights", expanded=True):
            field("Hot Metal Weight", "hot_metal_weight")
            field("Scrap", "scrap")

        with st.expander("Blow & Oxygen", expanded=True):
            field("Blow Duration", "blow_duration")
            field("Oxygen Volume", "O2")

        with st.expander("Additions", expanded=True):
            field("Lime", "lime")
            field("Dolomite", "dolo")
            field("Sinter", "sinter")
            field("Iron Ore", "iron_ore")

        with st.expander("Slag / End-Point Indicators", expanded=True):
            field("Basicity", "basicity")
            field("FeO in Slag", "FeO")

        st.session_state.payload = payload

    with right:
        st.subheader("Prediction Output")
        t0 = time.time()
        yhat = predict_temp(st.session_state.payload)
        ms = (time.time() - t0) * 1000.0

        st.metric("Predicted Tap Temperature (°C)", f"{yhat:.1f}")
        st.caption(f"Latency: {ms:.0f} ms")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save Snapshot"):
                snap = dict(st.session_state.payload)
                snap["_predicted_temp_C"] = float(yhat)
                snap["_timestamp"] = pd.Timestamp.utcnow().isoformat()
                st.download_button("Download Snapshot JSON", data=json.dumps(snap, indent=2), file_name="bof_snapshot.json", mime="application/json")
        with c2:
            if st.button("Reset to Mid-Ranges"):
                st.session_state.payload = build_default_input()
                st.rerun()

        st.divider()
        st.caption(f"Model artifact: artifacts/bof_temp_model.keras")
        st.caption(f"Feature contract: 15 features")

with tab2:
    st.subheader("Local Sensitivity Contribution (around current operating point)")
    base, dfc = sensitivity_contrib(st.session_state.payload, frac=0.01)
    st.metric("Base Prediction (°C)", f"{base:.1f}")
    st.dataframe(dfc, use_container_width=True, height=420)
    st.bar_chart(dfc.set_index("feature")["delta_temp_C"])

with tab3:
    st.subheader("API View (Request / Response)")
    req = dict(st.session_state.payload)
    resp_base, resp_contrib = sensitivity_contrib(st.session_state.payload, frac=0.01)
    response = {
        "predicted_temp_C": float(resp_base),
        "top_contributors": resp_contrib.head(5).to_dict(orient="records"),
        "model_artifact": "artifacts/bof_temp_model.keras",
        "feature_count": len(FEATURES),
        "transforms": schema.get("transforms", {})
    }
    c1, c2 = st.columns(2)
    with c1:
        st.code(json.dumps(req, indent=2), language="json")
    with c2:
        st.code(json.dumps(response, indent=2), language="json")
