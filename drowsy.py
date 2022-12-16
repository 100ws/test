#!/usr/bin/env python3

from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
from netifaces import interfaces, ifaddresses, AF_INET
for ifaceName in interfaces():
    addresses = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr':'No IP addr'}] )]
    #print ('%s: %s' % (ifaceName, ', '.join(addresses)))
import sys

adresses=str(addresses)[2:-2]
print(adresses)
import os

# import os

# with open("file_running.text","w") as write_data:
#     write_data.write("/home/pi/Desktop/final_drowsiness/drowsy.py")


app = Flask(__name__)

video_camera = None
global_frame = None

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/')
def hello_world():
    ip_addr = request.remote_addr
    return '<h1> Your IP address is:' + ip_addr
    

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")

def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        with open("/home/pi/Desktop/final_drowsiness/reading_data.text","r") as reading_data:
            data=reading_data.read()
            data=int(data)
            print(data)
        if data==1:
            
            frame = video_camera.get_frame()

            if frame != None:
                global_frame = frame
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
        elif data==0:
            # sys.exit()
            os.system("pkill -f /home/pi/Desktop/final_drowsiness/drowsy.py")

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host=adresses, threaded=True)
