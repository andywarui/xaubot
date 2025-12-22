"""
Streamlit Dashboard for MT5 EA Monitoring
Real-time visualization of model performance and trading metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime, timedelta


st.set_page_config(
    page_title="XAUUSD Neural Bot Monitor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 XAUUSD Neural Bot - Live Performance Dashboard")
st.markdown("---")


def load_prediction_log(log_file="prediction_log.csv"):
    """Load prediction log"""
    try:
        possible_paths = [
            Path("MQL5/Files") / log_file,
            Path(".") / log_file,
        ]

        for path in possible_paths:
            if path.exists():
                df = pd.read_csv(path, sep=';')
                df['time'] = pd.to_datetime(df['time'])
                return df

        return None
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return None


def load_monitoring_results():
    """Load monitoring results JSON"""
    results_file = Path("monitoring_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


# Sidebar
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 10, 300, 60)
lookback_hours = st.sidebar.slider("Lookback Period (hours)", 1, 168, 24)

# Auto-refresh
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")

# Load data
pred_df = load_prediction_log()
monitoring_results = load_monitoring_results()

if pred_df is not None and len(pred_df) > 0:
    st.sidebar.success(f"✅ {len(pred_df)} predictions loaded")

    # Filter by lookback period
    cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
    pred_df = pred_df[pred_df['time'] >= cutoff_time]

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", len(pred_df))

    with col2:
        avg_confidence = pred_df['best_prob'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")

    with col3:
        long_pct = (pred_df['best_class'] == 2).sum() / len(pred_df)
        st.metric("LONG Signals", f"{long_pct:.1%}")

    with col4:
        short_pct = (pred_df['best_class'] == 0).sum() / len(pred_df)
        st.metric("SHORT Signals", f"{short_pct:.1%}")

    # Prediction distribution over time
    st.subheader("📊 Prediction Distribution Over Time")

    fig = go.Figure()

    # Resample to hourly
    pred_df_hourly = pred_df.set_index('time').resample('1H').agg({
        'p_long': 'mean',
        'p_hold': 'mean',
        'p_short': 'mean'
    }).reset_index()

    fig.add_trace(go.Scatter(
        x=pred_df_hourly['time'],
        y=pred_df_hourly['p_long'],
        name='LONG Probability',
        line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pred_df_hourly['time'],
        y=pred_df_hourly['p_hold'],
        name='HOLD Probability',
        line=dict(color='gray', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pred_df_hourly['time'],
        y=pred_df_hourly['p_short'],
        name='SHORT Probability',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Probability",
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Prediction class distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Prediction Class Distribution")

        class_counts = pred_df['best_class'].value_counts().sort_index()
        class_names = ['SHORT', 'HOLD', 'LONG']

        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=[class_counts.get(i, 0) for i in range(3)],
                marker_color=['red', 'gray', 'green']
            )
        ])

        fig.update_layout(
            xaxis_title="Signal Type",
            yaxis_title="Count",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Confidence Distribution")

        fig = go.Figure(data=[
            go.Histogram(
                x=pred_df['best_prob'],
                nbinsx=20,
                marker_color='blue'
            )
        ])

        fig.update_layout(
            xaxis_title="Confidence",
            yaxis_title="Frequency",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    # Recent predictions table
    st.subheader("🔍 Recent Predictions")

    recent = pred_df.tail(20).sort_values('time', ascending=False)
    recent['Signal'] = recent['best_class'].map({0: 'SHORT', 1: 'HOLD', 2: 'LONG'})
    recent['Confidence'] = recent['best_prob'].apply(lambda x: f"{x:.1%}")

    display_df = recent[['time', 'Signal', 'Confidence', 'p_long', 'p_hold', 'p_short']].copy()
    display_df['p_long'] = display_df['p_long'].apply(lambda x: f"{x:.3f}")
    display_df['p_hold'] = display_df['p_hold'].apply(lambda x: f"{x:.3f}")
    display_df['p_short'] = display_df['p_short'].apply(lambda x: f"{x:.3f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Monitoring results (if available)
    if monitoring_results:
        st.subheader("🎯 Model Accuracy (from monitoring)")

        latest_metrics = monitoring_results[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Overall Accuracy", f"{latest_metrics['overall_accuracy']:.1%}")

        with col2:
            st.metric("LONG Accuracy", f"{latest_metrics['class_accuracies']['LONG']:.1%}")

        with col3:
            st.metric("HIGH Conf. Accuracy", f"{latest_metrics['high_confidence_accuracy']:.1%}")

        # Accuracy trend
        if len(monitoring_results) > 1:
            st.subheader("📉 Accuracy Trend")

            accuracy_df = pd.DataFrame([
                {
                    'time': r['timestamp'],
                    'accuracy': r['overall_accuracy']
                }
                for r in monitoring_results
            ])
            accuracy_df['time'] = pd.to_datetime(accuracy_df['time'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=accuracy_df['time'],
                y=accuracy_df['accuracy'],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                name='Accuracy'
            ))

            # Add threshold line
            fig.add_hline(y=0.55, line_dash="dash", line_color="red",
                          annotation_text="Min Acceptable (55%)")

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Accuracy",
                yaxis_tickformat='.0%',
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

else:
    st.sidebar.warning("⚠️ No prediction data found")
    st.info("👉 Make sure the EA is running and generating predictions.")
    st.info("📁 Prediction log should be in: MQL5/Files/prediction_log.csv")

# Footer
st.markdown("---")
st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("*Dashboard will auto-refresh based on sidebar settings*")

# Auto-refresh
if refresh_rate > 0:
    import time
    time.sleep(refresh_rate)
    st.rerun()
