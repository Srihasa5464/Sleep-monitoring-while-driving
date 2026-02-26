from flask import Flask, request, jsonify, render_template, send_from_directory
import base64
import os
from datetime import datetime
app = Flask(__name__)
# Folder for saved snapshots
SNAPSHOT_FOLDER = os.path.join("static", "snapshots")
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
# Store recent events in memory
recent_events = []
@app.route("/")
def dashboard():
    return render_template("ui.html")
@app.route("/alert", methods=["POST"])
def alert():
    """Receive alert data and optional image from client"""
    data = request.get_json()
    timestamp = data.get("timestamp")
    event_type = data.get("event")
    location = data.get("location", "Unknown")


    print(f"[{timestamp}] ALERT: {event_type} from {location}")

    filename = None
    img_b64 = data.get("image")
    if img_b64:
        img_data = base64.b64decode(img_b64)
        filename = f"{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(SNAPSHOT_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(img_data)
        print(f"Image saved: {filepath}")

    # Add event to memory
    recent_events.insert(0, {
        "timestamp": timestamp,
        "event": event_type,
        "location": location,
        "filename": filename
    })
    recent_events[:] = recent_events[:20]  # Keep latest 20 events

    return jsonify({"status": "ok"}), 200

@app.route("/snapshots_list")
def snapshots_list():
    """Return list of snapshots & events"""
    return jsonify(recent_events)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
