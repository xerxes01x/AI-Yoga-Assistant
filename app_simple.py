from flask import Flask, render_template, Response
import numpy as np
import cv2
import time 
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import data as data

app = Flask(__name__)

# Load the model
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

# Initialize webcam with error handling
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open webcam. Running in headless mode.")
        cap = None
except Exception as e:
    print(f"Warning: Webcam initialization failed: {e}")
    cap = None

dataList = data.AngleData

# EDGES for pose connections
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def compare_right_arm(right_arm):
    tadasan = [y for x, y in list(dataList[0].items()) if type(y) == int]
    
    if(right_arm <= tadasan[0]):
        acc = (right_arm/tadasan[0])*100
    else:
        acc = 0
        
    if abs(tadasan[0]-right_arm) <= 10:
        print("Your right arm is accurate")
    else:
        print("Your right arm is not accurate")

    return acc

def compare_left_arm(left_arm):
    tadasan = [y for x, y in list(dataList[0].items()) if type(y) == int]
        
    if(left_arm <= tadasan[1]):
        acc = (left_arm/tadasan[1])*100
    else:
        acc = 0
        
    if abs(tadasan[1]-left_arm) <= 10:    
        print("Your left arm is accurate")
    else:
        print("Your left arm is not accurate, try again")
    
    return acc
    
def compare_right_leg(right_leg):
    tadasan = [y for x, y in list(dataList[0].items()) if type(y) == int]

    if(right_leg <= tadasan[2]):
        acc = (right_leg/tadasan[2])*100
    else:
        acc = 0

    if abs(tadasan[2]-right_leg) <= 10:
        print("Your right leg is accurate")                
    else:
        print("Your right leg is not accurate, try again") 

    return acc
        
def compare_left_leg(left_leg):
    tadasan = [y for x, y in list(dataList[0].items()) if type(y) == int]
    
    if(left_leg <= tadasan[3]):
        acc = (left_leg/tadasan[3])*100
    else:
        acc = 0

    if abs(tadasan[3]-left_leg and left_leg < tadasan[3]) <= 10:
        print("Your left leg is accurate") 
    else:
        print("Your left leg is not accurate, try again") 
    
    return acc

arr = np.array([])
    
def generate_frames(arr):
    if cap is None:
        # Return a placeholder frame if no webcam is available
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No webcam available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    count = 0
    timeout = 20
    timeout_start = time.time()
    
    while time.time() < timeout_start + timeout:
        while True:
            # Read the camera frame
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
      
            # Resize the image
            img = frame.copy()
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
            input_img = tf.cast(img, dtype=tf.int32)
                
            # Detect the image
            results = movenet(input_img)
            keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
            
            # Overlay keypoints (no GUI window in server)
            loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)

            # Simple pose analysis (without MediaPipe)
            # For now, we'll just show the frame with keypoints
            # In a full implementation, you'd need MediaPipe for detailed pose detection
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def accuracyCalculation(arr):
    accArray = np.array([])
    sum = 0
    
    for j in range(0, len(arr)-1, 4):
        for i in range(j, j+4):
            print("arr[i]", arr[i])
            sum = sum + arr[i]
        accur = sum/4
        accArray = np.append(accArray, accur/4)
    
    return accArray

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
    accArray = accuracyCalculation(arr)
    values = [12, 19, 3, 5, 2, 3]
    labels = ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange']
    colors = ['#ff0000','#0000ff','#ffffe0','#008000','#800080','#FFA500', '#FF2554']
    return render_template('charts.html', values=accArray, labels=labels, colors=colors)

@app.route('/video')
def video():
    return Response(generate_frames(arr), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀 Starting AI Yoga Assistant...")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    print("⚠️  Note: Full pose detection requires MediaPipe (not compatible with Python 3.13)")
    print("💡 For full functionality, use Python 3.9-3.11 with: pip install mediapipe")
    app.run(host="127.0.0.1", debug=True)

