from flask import Flask, render_template, Response, jsonify
import cv2, time, os, threading, numpy as np
from ultralytics import YOLO
app = Flask(__name__)
face_model = YOLO("yolov8m-face_openvino_model", task="detect")
face_model(np.zeros((320,320,3), dtype=np.uint8), imgsz=320, verbose=False)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640); cap.set(4, 480)
os.makedirs("snapshots", exist_ok=True)
camera_active, latest_raw, latest_frame, blur_mode = False, None, None, False
stats = {"faces": 0, "fps": 0}
def cam_loop():
    global latest_raw
    while True:
        ok, fr = cap.read()
        if ok: latest_raw = fr
        else: time.sleep(0.01)
threading.Thread(target=cam_loop, daemon=True).start()
def hud(img, x1, y1, x2, y2, cf):
    c, l = (255,204,0), int((x2-x1)*0.22)
    for a,b in [((x1,y1),(x1+l,y1)),((x1,y1),(x1,y1+l)),((x2,y1),(x2-l,y1)),((x2,y1),(x2,y1+l)),((x1,y2),(x1+l,y2)),((x1,y2),(x1,y2-l)),((x2,y2),(x2-l,y2)),((x2,y2),(x2,y2-l))]: cv2.line(img,a,b,c,3)
    cv2.putText(img, f"ID:{int(cf*100)}%", (x1,y1-10), 0, 0.6, c, 2)
def gen():
    global latest_frame, stats
    pt = time.time()
    while True:
        if not camera_active or latest_raw is None:
            stats = {"faces":0,"fps":0}; time.sleep(0.05); continue
        fr = latest_raw.copy()
        res = face_model(fr, imgsz=320, verbose=False)
        bxs = res[0].boxes; stats["faces"] = len(bxs)
        for b in bxs:
            x1,y1,x2,y2 = map(int, b.xyxy[0]); cf = float(b.conf[0])
            if blur_mode:
                r = fr[max(0,y1):y2, max(0,x1):x2]
                if r.size>0: fr[max(0,y1):y2, max(0,x1):x2] = cv2.GaussianBlur(r,(51,51),30)
            hud(fr,x1,y1,x2,y2,cf)
        ct = time.time(); stats["fps"] = int(1/max(ct-pt,0.001)); pt = ct
        latest_frame = fr.copy()
        _, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
@app.route("/")
def index(): return render_template("index.html")
@app.route("/video_feed")
def video_feed(): return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/toggle_camera", methods=["POST"])
def tog_cam():
    global camera_active
    camera_active = not camera_active
    return jsonify({"active": camera_active})
@app.route("/stats")
def get_st(): return jsonify(stats)
@app.route("/toggle_blur", methods=["POST"])
def tog_bl():
    global blur_mode
    blur_mode = not blur_mode
    return jsonify({"blur": blur_mode})
@app.route("/snapshot", methods=["POST"])
def snap():
    if latest_frame is not None and camera_active:
        fn = f"snapshots/face{int(time.time())}.jpg"; cv2.imwrite(fn, latest_frame)
        return jsonify({"saved": fn})
    return jsonify({"saved": None})
if __name__ == "__main__": import webbrowser, threading; threading.Timer(0.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start(); app.run(debug=False, threaded=True)