import streamlit as str
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from pipeline import ClassroomMonitorPipeline
from database import init_db, get_classroom_summary, get_student_history

str.set_page_config(page_title="AI Classroom Behaviour Dashboard", layout="wide")
init_db()

@str.cache_resource
def load_pipeline():
    return ClassroomMonitorPipeline()

pipeline = load_pipeline()

str.title("🎓 AI Classroom Behaviour Monitoring System")
str.markdown("---")

col_feed, col_analytics = str.columns([2, 1])

with col_feed:
    str.subheader("📹 Live Monitoring Matrix")
    run_feed = str.checkbox("Start Live Stream Camera Engine", value=False)
    frame_window = str.image([])
    
    if run_feed:
        camera = cv2.VideoCapture(0)
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                str.error("Failed to receive stream from camera device.")
                break
                
            processed_frame = pipeline.process_frame(frame)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB")
        camera.release()

with col_analytics:
    str.subheader("📊 Macro Classroom Insights")
    summary = get_classroom_summary()
    
    str.metric(label="Detected Active Students", value=summary["active_students"])
    str.metric(label="Average Classroom Attention Index", value=f"{summary['avg_attention']}%")
    
    violations = pd.DataFrame({
        'Incident Class': ['Drowsy', 'Distracted', 'Phone Violation'],
        'Occurrences': [summary["total_drowsy_incidents"], summary["total_distracted_incidents"], summary["total_phone_violations"]]
    })
    fig = px.bar(violations, x='Incident Class', y='Occurrences', title="Classroom Incident Log distribution")
    str.plotly_chart(fig, use_container_width=True)

    str.markdown("---")
    str.subheader("🔍 Single Student Profiler Look-Up")
    lookup_id = str.number_input("Provide Student Tracking ID", min_value=1, value=1, step=1)
    
    if str.button("Pull Analytics History Ledger"):
        history = get_student_history(lookup_id)
        if history:
            df = pd.DataFrame(history, columns=["Timestamp", "Attention Score", "Drowsy", "Looking Away", "Phone Usage"])
            str.dataframe(df, use_container_width=True)
            fig_trend = px.line(df, x="Timestamp", y="Attention Score", title=f"Attention Trend Index Timeline: Student {lookup_id}")
            str.plotly_chart(fig_trend, use_container_width=True)
        else:
            str.warning(f"No logged ledger history entries discovered for Track ID: {lookup_id}")