import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sqlalchemy import func
from Storage.database import Session
from Storage.cow_instance import Detection
import matplotlib.ticker as mticker

########################################
# Data Loading and Aggregation Functions
########################################

def load_detection_data_sample(sample_fraction=0.1) -> pd.DataFrame:
    session = Session()
    try:
        detections = session.query(Detection).all()
        data = [{
            "tracklet_id": d.tracklet_id,
            "timestamp": d.timestamp,
            "posture": d.posture
        } for d in detections]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values("timestamp", inplace=True)
        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42)
        return df
    finally:
        session.close()

def load_aggregated_detection_data() -> pd.DataFrame:
    session = Session()
    try:
        results = session.query(
            Detection.tracklet_id,
            Detection.posture,
            func.count(Detection.id).label("count")
        ).group_by(Detection.tracklet_id, Detection.posture).all()
        data = []
        for tracklet_id, posture, count in results:
            data.append({
                "tracklet_id": tracklet_id,
                "posture": posture,
                "count": count
            })
        df = pd.DataFrame(data)
        return df
    finally:
        session.close()

def aggregate_by_tracklet(agg_df: pd.DataFrame) -> pd.DataFrame:
    grouped = agg_df.pivot(index="tracklet_id", columns="posture", values="count").fillna(0)
    if "Standing" not in grouped.columns:
        grouped["Standing"] = 0
    if "Lying" not in grouped.columns:
        grouped["Lying"] = 0
    grouped["Total"] = grouped["Standing"] + grouped["Lying"]
    grouped["Standing %"] = (grouped["Standing"] / grouped["Total"]) * 100
    grouped["Lying %"] = (grouped["Lying"] / grouped["Total"]) * 100
    return grouped.reset_index()

def aggregate_total(summary_df: pd.DataFrame) -> dict:
    total_standing = summary_df["Standing"].sum()
    total_lying = summary_df["Lying"].sum()
    total = total_standing + total_lying
    standing_pct = (total_standing / total) * 100 if total > 0 else 0
    lying_pct = (total_lying / total) * 100 if total > 0 else 0
    return {
        "Total Standing": total_standing,
        "Total Lying": total_lying,
        "Standing %": standing_pct,
        "Lying %": lying_pct,
        "Total Detections": total
    }

########################################
# Visualization in Tkinter with Scrollable Table
########################################

def visualize_in_tk():
    # Load data
    df_sample = load_detection_data_sample(sample_fraction=0.05)
    if df_sample.empty:
        print("No detection data found for scatter plot.")
        return
    agg_df = load_aggregated_detection_data()
    summary_df = aggregate_by_tracklet(agg_df)
    total_summary = aggregate_total(summary_df)

    # Create a Matplotlib figure for the scatter plot with improved styling
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"Standing": "red", "Lying": "purple"}
    for _, row in df_sample.iterrows():
        # Convert tracklet_id to int if necessary:
        tracklet_id = int(row["tracklet_id"])
        color = colors.get(row["posture"], "blue")
        ax.scatter(row["timestamp"], tracklet_id, color=color, s=15, alpha=0.7)
        
    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel("Tracklet ID", fontsize=12)
    ax.set_title("Detection Events Over Time", fontsize=14)
    ax.grid(True)
    # Force y-axis to have integer ticks only
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Create a Tkinter window with a nice style
    root = tk.Tk()
    root.title("Detection Visualization")
    style = ttk.Style(root)
    style.theme_use('clam')

    # Create a PanedWindow to divide left (scatter plot) and right (table)
    paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
    paned.pack(fill=tk.BOTH, expand=True)

    # Left frame for the scatter plot
    left_frame = tk.Frame(paned, bg="white")
    paned.add(left_frame, stretch="always")
    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Right frame for the summary table with a scrollbar
    right_frame = tk.Frame(paned)
    paned.add(right_frame, stretch="always")
    tree_scroll = tk.Scrollbar(right_frame)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree = ttk.Treeview(right_frame, yscrollcommand=tree_scroll.set)
    tree_scroll.config(command=tree.yview)
    tree["columns"] = ("Standing", "Standing %", "Lying", "Lying %", "Total")
    tree.column("#0", width=100, minwidth=100, anchor=tk.CENTER)
    tree.column("Standing", anchor=tk.CENTER, width=80)
    tree.column("Standing %", anchor=tk.CENTER, width=80)
    tree.column("Lying", anchor=tk.CENTER, width=80)
    tree.column("Lying %", anchor=tk.CENTER, width=80)
    tree.column("Total", anchor=tk.CENTER, width=80)
    tree.heading("#0", text="Tracklet", anchor=tk.CENTER)
    tree.heading("Standing", text="Standing", anchor=tk.CENTER)
    tree.heading("Standing %", text="Standing %", anchor=tk.CENTER)
    tree.heading("Lying", text="Lying", anchor=tk.CENTER)
    tree.heading("Lying %", text="Lying %", anchor=tk.CENTER)
    tree.heading("Total", text="Total", anchor=tk.CENTER)
    for _, row in summary_df.iterrows():
        tracklet = row["tracklet_id"]
        standing = int(row["Standing"])
        standing_pct = f"{row['Standing %']:.1f}%"
        lying = int(row["Lying"])
        lying_pct = f"{row['Lying %']:.1f}%"
        total = int(row["Total"])
        tree.insert("", tk.END, text=str(tracklet),
                    values=(standing, standing_pct, lying, lying_pct, total))
    tree.insert("", tk.END, text="TOTAL",
                values=(total_summary["Total Standing"],
                        f'{total_summary["Standing %"]:.1f}%',
                        total_summary["Total Lying"],
                        f'{total_summary["Lying %"]:.1f}%',
                        total_summary["Total Detections"]))
    tree.pack(fill=tk.BOTH, expand=True)

    def on_closing():
        from Storage.database import engine
        # Dispose of the engine to close pooled connections
        engine.dispose()
        # Quit and destroy the Tkinter root
        root.quit()
        root.destroy()
        # Force the process to exit
        import sys
        sys.exit(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    visualize_in_tk()
