

import streamlit as st
from transformers import pipeline
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import time
import torch

print("[INIT] Streamlit app starting up…")

# -------------------------------
# Configuration
# -------------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # Hugging Face model
NEUTRAL_THRESHOLD = 0.60  # if top score < threshold -> consider neutral
print(f"[CONFIG] MODEL_NAME={MODEL_NAME}, NEUTRAL_THRESHOLD={NEUTRAL_THRESHOLD}")

# -------------------------------
# Caching the pipeline to speed up repeated calls
# -------------------------------
@st.cache_resource
def load_pipeline(model_name: str):
    print(f"[PIPELINE] Loading HugFace pipeline for {model_name}")
    # detect CUDA availability and choose device accordingly (0 -> first GPU, -1 -> CPU)
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        print(f"[PIPELINE] torch.cuda.is_available() check failed: {e}")
        cuda_available = False
    device = 0 if cuda_available else -1
    print(f"[PIPELINE] CUDA available={cuda_available}; using device={device}")
    return pipeline("sentiment-analysis", model=model_name, device=device)

print("[PIPELINE] Loading model…")
pipe = load_pipeline(MODEL_NAME)
print("[PIPELINE] Model loaded.")

# -------------------------------
# Utility: analyze single text
# -------------------------------
def analyze_text(text: str):
    """Return a dict: {label, score, normalized_label}
    normalized_label: one of 'Positive', 'Negative', 'Neutral' (adds Neutral handling)
    """
    print(f"[ANALYZE] Received text: {text}")
    text = text.strip()
    if not text:
        print("[ANALYZE] Empty text received. Returning None.")
        return None

    # call HF pipeline
    print("[ANALYZE] Calling HF pipeline…")
    result = pipe(text, truncation=True)[0]
    print(f"[ANALYZE] Raw pipeline result: {result}")

    label = result["label"]  # typically 'POSITIVE' or 'NEGATIVE'
    score = float(result["score"])  # confidence for that label
    print(f"[ANALYZE] Extracted label={label}, score={score}")

    # convert to normalized label + compute percentages for visualization
    if score < NEUTRAL_THRESHOLD:
        print("[ANALYZE] Classified as NEUTRAL based on threshold.")
        normalized = "Neutral"
        neutral_share = 1.0 - score
        pos_share = score if label.upper() == "POSITIVE" else 0.0
        neg_share = score if label.upper() == "NEGATIVE" else 0.0
        total = neutral_share + pos_share + neg_share
        pos_pct = pos_share / total
        neg_pct = neg_share / total
        neu_pct = neutral_share / total
    else:
        print("[ANALYZE] Classified as NON-NEUTRAL.")
        normalized = "Positive" if label.upper() == "POSITIVE" else "Negative"
        pos_pct = score if label.upper() == "POSITIVE" else 0.0
        neg_pct = score if label.upper() == "NEGATIVE" else 0.0
        neu_pct = 1.0 - max(pos_pct, neg_pct)
        total = pos_pct + neg_pct + neu_pct
        pos_pct /= total
        neg_pct /= total
        neu_pct /= total

    print(f"[ANALYZE] Normalized label={normalized}, pos={pos_pct}, neu={neu_pct}, neg={neg_pct}")

    return {
        "raw_label": label,
        "score": score,
        "label": normalized,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "neu_pct": neu_pct,
    }

# -------------------------------
# Initialize session state for history
# -------------------------------
print("[SESSION] Checking for history state…")
if "history" not in st.session_state:
    print("[SESSION] Creating new history list.")
    st.session_state.history = []
else:
    print(f"[SESSION] Existing history length: {len(st.session_state.history)}")

# -------------------------------
# Streamlit UI
# -------------------------------
print("[UI] Building UI components…")
st.set_page_config(page_title="Real-Time Sentiment Analyzer", layout="wide")
st.title("Real‑Time Sentiment Analyzer")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Type text to analyze")

    def _on_change():
        print(f"[INPUT] on_change triggered. New input_text='{st.session_state.input_text}'")
        st.session_state.last_input = st.session_state.input_text

    if "last_input" not in st.session_state:
        print("[SESSION] Initializing last_input=''")
        st.session_state.last_input = ""

    st.text_area("Enter text", key="input_text", height=150, on_change=_on_change)

    analyze = st.button("Analyze")
    print(f"[INPUT] Analyze button pressed? {analyze}")

    input_text = st.session_state.get("last_input", "").strip()
    print(f"[INPUT] last_input='{input_text}'")

    auto_trigger = False
    if input_text:
        auto_trigger = analyze or (st.session_state.input_text != "" and st.session_state.input_text == input_text)
    print(f"[INPUT] auto_trigger={auto_trigger}")

    if analyze or (st.session_state.input_text and st.session_state.input_text == input_text):
        print("[INPUT] Triggering sentiment analysis…")
        result = analyze_text(input_text)
        if result is not None:
            print(f"[HISTORY] Adding new entry: {result}")
            entry = {
                "text": input_text,
                "result": result,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.history.insert(0, entry)
            st.success(f"Detected: {result['label']} (confidence {result['score']:.2f})")
        else:
            print("[HISTORY] analyze_text returned None (empty input)")

    st.write("\n")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Clear history"):
            print("[HISTORY] Clearing history…")
            st.session_state.history = []
            st.experimental_rerun()
    with c2:
        st.markdown("*Tip: type or paste text and press Analyze. The app caches the model to keep responses fast.*")

    if st.session_state.history:
        print("[UI] Displaying latest analysis…")
        latest = st.session_state.history[0]
        st.markdown("---")
        st.subheader("Latest analysis")
        st.write(f"**Text:** {latest['text']}")
        st.write(f"**Sentiment:** {latest['result']['label']} — confidence {latest['result']['score']:.2f}")
    else:
        print("[UI] No history to display yet.")

with col_right:
    st.subheader("Visualization")

    if st.session_state.history:
        print("[VIS] Building charts…")
        latest = st.session_state.history[0]['result']
        labels = ["Positive", "Neutral", "Negative"]
        values = [latest['pos_pct'], latest['neu_pct'], latest['neg_pct']]

        fig_pie = px.pie(values=values, names=labels, color=labels,
                         color_discrete_map={"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_latest")

        positivity = int(round(latest['pos_pct'] * 100))
        print(f"[VIS] Gauge positivity={positivity}")
        # Build a base gauge template (we'll animate the value)
        def make_gauge(val:int):
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=val,
                delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2ca02c"},
                    'steps': [
                        {'range': [0, 40], 'color': '#ffd6d6'},
                        {'range': [40, 60], 'color': '#f0f0f0'},
                        {'range': [60, 100], 'color': '#d6ffd6'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': val}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                              transition={'duration': 300, 'easing': 'cubic-in-out'})
            return fig

        # Animate the gauge from 0 to the target positivity in a few frames for a smooth effect
        placeholder = st.empty()
        steps = 8
        if positivity <= 5:
            frame_values = [positivity]
        else:
            step_size = max(1, int(positivity / steps))
            frame_values = list(range(0, positivity, step_size)) + [positivity]

        for val in frame_values:
            fig_frame = make_gauge(val)
            placeholder.plotly_chart(fig_frame, use_container_width=True, key=f"gauge_frame_{val}")
            # short sleep to allow perceived animation (keeps UI responsive)
            time.sleep(0.04)

        # final render (ensure exact final value)
        fig_final = make_gauge(positivity)
        placeholder.plotly_chart(fig_final, use_container_width=True, key=f"gauge_final_{positivity}")
    else:
        print("[VIS] No history — skipping charts.")

# -------------------------------
# Chat history panel (below)
# -------------------------------
st.markdown("---")
st.subheader("Analysis history")

if st.session_state.history:
    print(f"[HISTORY] Rendering history list (length={len(st.session_state.history)})")
    for i, entry in enumerate(st.session_state.history):
        with st.expander(f"{entry['time']} — {entry['result']['label']} (conf {entry['result']['score']:.2f})", expanded=(i==0)):
            st.write(entry['text'])
            labels = ["Positive", "Neutral", "Negative"]
            values = [entry['result']['pos_pct'], entry['result']['neu_pct'], entry['result']['neg_pct']]
            fig = px.pie(values=values, names=labels)
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, key=f"hist_pie_{i}")
else:
    print("[HISTORY] No history entries found.")
    st.write("No history yet.")

# Footer
print("[FOOTER] Rendering footer…")
st.markdown("---")
st.caption("Model: " + MODEL_NAME + "  • Neutral threshold: " + str(NEUTRAL_THRESHOLD))
