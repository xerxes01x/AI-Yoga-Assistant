from flask import Flask, render_template, Response
import numpy as np
import cv2
import time 

app = Flask(__name__)

# Don't initialize webcam on startup - do it when needed
cap = None

def get_webcam():
    global cap
    if cap is None:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Warning: Could not open webcam.")
                cap = None
        except Exception as e:
            print(f"Warning: Webcam initialization failed: {e}")
            cap = None
    return cap

def generate_frames():
    webcam = get_webcam()
    
    if webcam is None:
        # Return a placeholder frame if no webcam is available
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No webcam available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, "Connect a webcam for live video", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    while True:
        # Read the camera frame
        success, frame = webcam.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        
        # Add a simple overlay
        cv2.putText(frame, "AI Yoga Assistant - Web Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Install MediaPipe for pose detection", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/tracks')
def tracks():
    return render_template('tracks.html')

@app.route('/yoga')
def yoga():
    return render_template('yoga.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/charts')
def charts():
    # Mock data for charts
    values = [12, 19, 3, 5, 2, 3]
    labels = ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange']
    colors = ['#ff0000','#0000ff','#ffffe0','#008000','#800080','#FFA500', '#FF2554']
    return render_template('charts.html', values=values, labels=labels, colors=colors)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀 Starting AI Yoga Assistant...")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    print("⚠️  Note: This is a web interface without AI pose detection")
    print("💡 For full AI functionality:")
    print("   1. Use Python 3.9-3.11 (not 3.13)")
    print("   2. Install: pip install mediapipe tensorflow tensorflow-hub")
    print("   3. Run: python app.py")
    print("🌐 Server starting...")
    app.run(host="127.0.0.1", debug=True)

