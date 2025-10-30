import sys, threading, time, difflib, threading, traceback, os
print("Python executable:", sys.executable)
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2, time, math
import numpy as np
from tomlkit import key
import speech_recognition as sr
from googletrans import Translator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
from controller import Robot # type: ignore
from collections import deque
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

rcParams.update({'figure.autolayout': True})  # global fix for layout

def excepthook(exc_type, exc, tb):
    print("[UNCAUGHT EXCEPTION]", exc_type.__name__)
    traceback.print_exception(exc_type, exc, tb)
sys.excepthook = excepthook

landmarks = {
    "church": ("-19", "25"),
    "city hall": ("19", "10"),
    "library": ("19", "27"),
    "bookstore": ("19", "27"),
    "barbershop": ("26", "25"),
    "bakery": ("26", "42"),
    "computer store": ("26", "57"),
    "offices": ("26", "150"),
    "hotel": ("26", "222.5"),
    "subway": ("-26", "18"),
    "public toilet 2": ("-26", "47"),
    "theater": ("-26", "90"),
    "bus stop 1": ("-26", "154.5"),
    "auditorium": ("-26", "185"),
    "diner": ("-26", "207.5"),
    "museum": ("-26", "240"),
    "phone booth": ("-18", "5"),
    "hospital": ("-18", "41"),
    "momo": ("-18", "75"),
    "medical clinic": ("-18", "90"),
    "news stand": ("-18", "111"),
    "dentist": ("-18", "130"),
    "building under construction": ("18", "10"),
    "mall": ("18", "52"),
    "bus stop 0": ("18", "81"),
    "motel": ("18", "110"),
    "cinema": ("18", "140"),
    "windmill": ("2", "45"),
    "silo": ("0", "30"),
    "lake house": ("-0", "10"),
    "lake": ("-0", "10"),
    "warehouse": ("10", "45"),
    "gym": ("-10", "30"),
    "bus stop 2": ("-10", "58"),
    "farm": ("11", "25"),
    "old town": ("-11", "35"),
    "kiosk": ("-9", "42"),
    "university": ("-9", "110"),
    "sapienza": ("-9", "110"),
    "bus stop 3": ("9", "48"),
    "supermarket": ("-17", "35"),
    "kfc": ("-17", "53"),
    "snack stand": ("-17", "78"),
    "ffc": ("4", "25"),
    "post office": ("-25", "28"),
    "car wash": ("-25", "45"),
    "factory": ("-25", "77.5"),
    "bus stop 4": ("-25", "101"),
    "gas station": ("25", "274"),
    "tunnel": ("-20", "120"),
    "prison": ("20", "80"),
    "law court": ("20", "45"),
    "jail": ("20", "80"),
    "bank": ("20", "15"),
    "skyscraper": ("-20", "230"),
    "green house": ("17", "70"),
    "purple house": ("-3", "20"),
    "pharmacy": ("14", "35"),
    "yellow house": ("-14", "11"),
    "restaurant": ("-14", "28.5"),
    "bus stop 7": ("-6", "15"),
    "red house": ("-6", "31.5"),
    "bus stop 5": ("6", "10"),
    "black house": ("-5", "20"),
    "gnome": ("5", "10"),
    "white house": ("-13", "70"),
    "brown house": ("21", "10"),
    "fire station": ("21", "45"),
    "pink house": ("22", "20"),
    "police station": ("-21", "20"),
    "public toilet": ("-21", "57"),
    "blue house": ("-24", "14"),
    "purple apartments": ("-16", "77"),
    "bar": ("-16", "15"),
    "school": ("25", "20"),
    "green apartments": ("25", "60"),
    "white apartments": ("25", "145"),
    "landfill": ("25", "190"),
    "bus stop 6": ("25", "214.5"),
    "garage": ("25", "225"),
    "black apartments": ("-25", "135"),
    "blue apartments": ("-25", "165"),
    "red apartments": ("-25", "205"),
    "orange apartments": ("-25", "240"),
}

landmarks_coordinates = {
    "church": ("285", "-486"),
    "city hall": ("270", "-512"),
    "library": ("293", "-509"),
    "bookstore": ("293", "-509"),
    "barbershop": ("323", "-468"),
    "bakery": ("324", "-452"),
    "computer store": ("327", "-437"),
    "offices": ("362", "-350"),
    "hotel": ("297", "-289"),
    "subway": ("269", "-327"),
    "public toilet 2": ("295", "-316"),
    "theater": ("328", "-334"),
    "bus stop 1": ("331", "-380"),
    "auditorium": ("295", "-401"),
    "diner": ("307", "-430"),
    "museum": ("299", "-463"),
    "phone booth": ("260", "-481"),
    "hospital": ("269", "-444"),
    "momo": ("244", "-418"),
    "medical clinic": ("248", "-399"),
    "news stand": ("244", "-379"),
    "dentist": ("250", "-359"),
    "building under construction": ("217", "-344"),
    "mall": ("216.5", "-390"),
    "bus stop 0": ("232", "-419"),
    "motel": ("232", "-450"),
    "cinema": ("234", "-476"),
    "windmill": ("352", "-489"),
    "silo": ("391", "-434"),
    "lake house": ("419", "-434"),
    "warehouse": ("347", "-525"),
    "gym": ("308", "-527"),
    "bus stop 2": ("337", "-541"),
    "farm": ("372", "-601"),
    "old town": ("326", "-588"),
    "kiosk": ("324", "-643"),
    "university": ("261", "-624"),
    "bus stop 3": ("281", "-655"),
    "supermarket": ("195", "-514"),
    "kfc": ("223", "-504"),
    "snack stand": ("244", "-506"),
    "ffc": ("230", "-617"),
    "post office": ("79.5", "-536"),
    "car wash": ("86.6", "-557"),
    "factory": ("87.5", "-584"),
    "bus stop 4": ("105", "-603"),
    "gas station": ("86.", "-504"),
    "tunnel": ("123", "-401"),
    "prison": ("126", "-296"),
    "law court": ("184", "-288"),
    "bank": ("228", "-300"),
    "skyscraper": ("185", "-323"),
    "green house": ("186", "-483"),
    "purple house": ("150", "-491"),
    "pharmacy": ("126", "-504"),
    "yellow house": ("109", "-524"),
    "restaurant": ("140", "-530"),
    "bus stop 7": ("171", "-521"),
    "red house": ("171", "-545"),
    "bus stop 5": ("186", "-531"),
    "black house": ("177", "-573"),
    "gnome": ("200", "-572"),
    "white house": ("191", "-606"),
    "brown house": ("207", "-554"),
    "fire station": ("231", "-560"),
    "pink house": ("258", "-572"),
    "public toilet": ("194", "-535"),
    "police station": ("237", "-525"),
    "blue house": ("295", "-549"),
    "purple apartments": ("282", "-603"),
    "bar": ("217", "-589"),
    "green apartments": ("180", "-660"),
    "school": ("209", "-649"),
    "white apartments": ("133", "-632"),
    "landfill": ("125", "-587"),
    "bus stop 6": ("108", "-568"),
    "garage": ("113", "-549"),
    "black apartments": ("106", "-644"),
    "blue apartments": ("122", "-679"),
    "red apartments": ("159", "-695"),
    "orange apartments": ("205", "-688"),
}

landmark_count = len(set(landmarks.values()))
print(f"{landmark_count} {len(set(landmarks_coordinates.values()))} landamrks!")

# ----------------------------
#  Webots sensor controller
# ----------------------------
class WebotsSensors(threading.Thread):
    def __init__(self):
        super().__init__()
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.timestep = 1
        print(f"timestep: {self.timestep} ms")
        self.cam_front = self.robot.getDevice("camera_front")
        self.cam_back = self.robot.getDevice("camera_back")
        self.cam_third_person = self.robot.getDevice("camera_third_person")
        self.cam_driver = self.robot.getDevice("camera_driver")
        self.cam_right = self.robot.getDevice("camera_right")
        self.cam_left = self.robot.getDevice("camera_left")
        self.gps = self.robot.getDevice("gps")
        self.compass = self.robot.getDevice("compass")
        # self.accelerometer = self.robot.getDevice("accelerometer")
        # self.lidar = self.robot.getDevice("lidar")
        self.rangefinder = self.robot.getDevice("range-finder")
        self.radar_front = self.robot.getDevice("radar_front")
        self.radar_back = self.robot.getDevice("radar_back")

        for cam in [self.cam_front, self.cam_back, self.cam_right, self.cam_left, self.cam_third_person, self.cam_driver]:
            cam.enable(self.timestep)
        self.cam_front.recognitionEnable(self.timestep)
        self.rangefinder.enable(self.timestep)
        self.radar_front.enable(self.timestep)
        self.radar_back.enable(self.timestep)

        self.running = True
        self.latest_images = {"front": None, "back": None, "right": None, "left": None, "third_person": None, "rangefinder": None}

        self.gps.enable(self.timestep)
        self.compass.enable(self.timestep)
        # self.accelerometer.enable(self.timestep)
        self.latest_gps = [0.0, 0.0, 0.0]
        self.latest_speed = 0.0
        self.latest_compass = [0.0, 0.0, 0.0]
        self.latest_accel = 0.0

    def run(self):
        try:
            while self.running and self.robot.step(self.timestep) != -1:
                self.latest_images["front"] = self.cam_front.getImage()
                self.latest_images["back"] = self.cam_back.getImage()
                self.latest_images["right"] = self.cam_right.getImage()
                self.latest_images["left"] = self.cam_left.getImage()
                # self.latest_images["third_person"] = self.cam_third_person.getImage()
                self.latest_images["rangefinder"] = self.rangefinder.getRangeImage()

                # update other sensors
                self.latest_gps = self.gps.getValues()
                self.latest_speed = self.gps.getSpeed()
                self.latest_compass = self.compass.getValues()
        except Exception as e:
            print(f"[ERROR] WebotsSensors thread crashed: {e}")
            traceback.print_exc()

    def get_qimage(self, raw_img, cam):
        if raw_img is None:
            return None
        w, h = cam.getWidth(), cam.getHeight()
        img = np.frombuffer(raw_img, np.uint8).reshape((h, w, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        qimg = QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        return qimg

    def get_depthqimage(self, raw_img, cam, invert=False):
        """Convert Webots rangefinder float image directly to 8-bit RGB (no normalization)."""
        if raw_img is None or cam is None:
            return None

        w = int(cam.getWidth())
        h = int(cam.getHeight())

        arr = np.array(raw_img, dtype=np.float32).reshape((h, w))
        # Clip directly to 0–255, no normalization or scaling
        arr = np.clip(arr, 0, 255)
        img8 = arr.astype(np.uint8)

        # Replicate to 3 RGB channels for proper display
        rgb = np.dstack([img8] * 3)

        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        return qimg
            
# ----------------------------
#  Dashboard GUI
# ----------------------------
class Dashboard(QtWidgets.QWidget):
    def __init__(self, sensors):
        super().__init__()
        self.sensors = sensors
        global landmarks
        global landmarks_coordinates
        self.landmarks = landmarks
        self.landmarks_coordinates = landmarks_coordinates

        self.setWindowTitle("Autonomous Car Dashboard")
        layout = QtWidgets.QVBoxLayout(self)

        # Command section
        self.command_input = QtWidgets.QLineEdit()
        self.command_input.setPlaceholderText(f"Choose your destination among {landmark_count} different landmarks!")
        self.command_input.setFocus()                                # ensure focus
        self.send_button = QtWidgets.QPushButton("Enter")
        self.send_button.clicked.connect(self.send_command)
        self.mic_button = QtWidgets.QPushButton("🎙️ Speak")
        self.mic_button.clicked.connect(self.record_audio)
        self.status_label = QtWidgets.QLabel("Destination: not specified")
        self.status_label.setStyleSheet("font-weight: bold; color: #2E8B57;")

        # allow Enter key to send
        self.command_input.returnPressed.connect(self.send_command)

        cmd_layout = QtWidgets.QHBoxLayout()
        cmd_layout.addWidget(self.command_input)
        cmd_layout.addWidget(self.send_button)
        cmd_layout.addWidget(self.mic_button)
        layout.addLayout(cmd_layout)

        layout.addWidget(self.status_label)

        # Camera views
        # # Main front camera
        # self.cam_labels = {}
        # self.cam_labels["front"] = QtWidgets.QLabel()
        # self.cam_labels["front"].setFixedSize(400, 400)  # bigger
        # self.cam_labels["front"].setStyleSheet("background-color: gray;")

        # layout.addWidget(self.cam_labels["front"])

        # # side cameras
        # sidecams_layout = QtWidgets.QHBoxLayout()
        # for name in ["left", "tight"]:
        #     lbl = QtWidgets.QLabel()
        #     lbl.setFixedSize(160, 130)  # smaller
        #     lbl.setStyleSheet("background-color: gray;")
        #     self.cam_labels[name] = lbl
        #     sidecams_layout.addWidget(lbl)
        # layout.addLayout(sidecams_layout)

        # depth_layout = QtWidgets.QHBoxLayout()
        # for name in ["back", "rangefinder"]:
        #     lbl = QtWidgets.QLabel()
        #     lbl.setFixedSize(160, 130)  # smaller
        #     lbl.setStyleSheet("background-color: gray;")
        #     self.cam_labels[name] = lbl
        #     depth_layout.addWidget(lbl)
        # layout.addLayout(depth_layout)

        cams_layout = QtWidgets.QHBoxLayout()

        # Left: front camera
        self.cam_labels = {}
        self.cam_labels["front"] = QtWidgets.QLabel()
        self.cam_labels["front"].setFixedSize(405, 405)
        self.cam_labels["front"].setStyleSheet("background-color: gray;")
        self.cam_labels["front"].setToolTip("Front Camera View")
        cams_layout.addWidget(self.cam_labels["front"])

        # Right: grid of 4 smaller cameras
        right_grid = QtWidgets.QGridLayout()
        small_names = ["left", "right", "back", "rangefinder"]
        for i, name in enumerate(small_names):
            lbl = QtWidgets.QLabel()
            lbl.setFixedSize(200, 200)
            lbl.setStyleSheet("background-color: gray;")
            self.cam_labels[name] = lbl
            self.cam_labels[name].setToolTip(f"{name.capitalize()} Camera View")
            row = i // 2
            col = i % 2
            right_grid.addWidget(lbl, row, col)
        cams_layout.addLayout(right_grid)

        layout.addLayout(cams_layout)

        # Other Sensors
        sensors_layout = QtWidgets.QHBoxLayout()

        # Compass widget
        self.compass_widget = CompassWidget(sensors)
        self.compass_widget.setToolTip("Compass")
        self.compass_widget.setFixedSize(120, 120)
        # sensors_layout.addWidget(self.compass_widget)

        # Accelerometer pedals
        self.pedals_widget = PedalsWidget(max_acc=5.0)
        self.pedals_widget.setToolTip("Acceleration Pedals")
        self.pedals_widget.setFixedSize(120, 120)
        # sensors_layout.addWidget(self.pedals_widget)

        # Gauge
        self.speed_widget = SpeedometerWidget()
        self.speed_widget.setToolTip("Speedometer")
        self.speed_widget.setFixedSize(120, 120)
        # sensors_layout.addWidget(self.speed_widget)

        # Radar
        self.radar_widget = RadarWidget(self.sensors.radar_front, self.sensors.radar_back)
        self.speed_widget.setFixedSize(120, 120)
        sensors_layout.addWidget(self.radar_widget)

        # Speed history
        speed_secs = 180
        self.speed_history = SpeedHistoryWidget(window_seconds=speed_secs, max_speed=20)  # adjust max_speed
        self.speed_history.setToolTip(f"Speed History - {speed_secs} seconds")
        self.speed_history.setFixedSize(125*4, 120)
        # sensors_layout.addWidget(self.speed_history)  # place where you want it in the layout

        # layout.addLayout(sensors_layout)

        # Left grid: other sensors
        # left_grid = QtWidgets.QGridLayout()
        # sensor_names = ["compass", "pedals", "speed", "speed_history"]
        # for i, widget in enumerate([self.compass_widget, self.pedals_widget, self.speed_widget, self.speed_history]):
        #     row = i // 3
        #     col = i % 3
        #     print(i, widget, row, col)
        #     left_grid.addWidget(widget, row, col)
        # sensors_layout.addLayout(left_grid)

        # Mini-map widget
        # self.map_widget = MiniMapWidget(
        #     map_path=r"map.png",
        #     world_bounds=((10, 505), (-285, -685))  # (min_x, max_x), (min_y, max_y) - adjust to match Webots GPS extents
        #     # (10, 505), (-285, -685)
        # )
        self.map_widget = MiniMapWidget("map.png", world_bounds=((15,495),(-270,-710)),
                     parent=self, landmarks_nav=landmarks, landmarks_xy=landmarks_coordinates)
        self.map_widget.setToolTip("GPS Map")
        # layout.addWidget(self.map_widget)

        # Right: mini-map
        # sensors_layout.addWidget(self.map_widget)
        # layout.addLayout(sensors_layout)

        # Left side layout
        left_layout = QtWidgets.QVBoxLayout()

        # Top row: three sensors
        top_row = QtWidgets.QHBoxLayout()
        for widget in [self.compass_widget, self.pedals_widget, self.speed_widget, self.radar_widget]:
            top_row.addWidget(widget)
        left_layout.addLayout(top_row)

        # Bottom row: speed history fills width
        self.speed_history.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        left_layout.addWidget(self.speed_history)

        # Right side: mini-map
        sensors_layout = QtWidgets.QHBoxLayout()
        sensors_layout.addLayout(left_layout)
        sensors_layout.addWidget(self.map_widget)

        layout.addLayout(sensors_layout)


        # Sensors text values
        self.gps_label = QtWidgets.QLabel("GPS: N/A")
        self.speed_label = QtWidgets.QLabel("Speed: N/A")
        self.compass_label = QtWidgets.QLabel("Compass: N/A")
        self.accel_label = QtWidgets.QLabel("Accel: N/A")
        # layout.addWidget(self.gps_label)
        # layout.addWidget(self.speed_label)
        # layout.addWidget(self.compass_label)
        # layout.addWidget(self.accel_label)

        # Update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(20)  # 50 FPS

    def extract_landmark_from_text(self, text, landmarks_dict):
        """
        Extracts the most likely landmark name from a free-form text.
        """
        text = text.lower()
        # exact match first
        for key in landmarks_dict.keys():
            if key in text:
                return key
        
        # fuzzy match as fallback
        closest = difflib.get_close_matches(text, landmarks_dict.keys(), n=1, cutoff=0.6)
        if closest:
            return closest[0]

        # default if nothing matches
        return "yellow house"

    def send_command(self):        
        user_input = self.command_input.text().strip()
        translator = Translator()
        user_input = translator.translate(user_input, src='it', dest='en').text

        if not user_input:
            print("[WARN] empty command")
            return
        
        key = user_input.lower()
        default_landmark = self.landmarks.get("yellow house")
        # dest = self.landmarks.get(user_input.lower(), default_landmark)
        key = self.extract_landmark_from_text(user_input, self.landmarks)
        dest = self.landmarks.get(key, default_landmark)

        if not dest:
            return

        # Optional: try fuzzy-matching if not exact match
        if dest is default_landmark and user_input.lower() not in self.landmarks:
            # find close keys
            close = difflib.get_close_matches(user_input.lower(), self.landmarks.keys(), n=1, cutoff=0.6)
            if close:
                matched_key = close[0]
                print(f"[INFO] Did you mean '{matched_key}'? Using it.")
                dest = self.landmarks[matched_key]
                key = matched_key

        # dest should be a tuple like ("-19", "25")
        if not isinstance(dest, (list, tuple)) or len(dest) < 2:
            print("[ERROR] landmark entry malformed:", dest)
            return

        lane, stop_pos = dest[0], dest[1]
        try:
            with open("new_dest.txt", "w") as f:
                f.write(f"{lane} {stop_pos}")
            print(f"[OK] Command sent: {user_input} -> lane {lane}, stop {stop_pos}")
            self.command_input.clear()
            self.status_label.setText(f"Destination: {key.capitalize()}")
        except Exception as e:
            print("[ERROR] writing new_dest.txt:", e)

    def update_ui(self):
        for name, cam in [
            ("front", self.sensors.cam_front),
            ("right", self.sensors.cam_right),
            ("left", self.sensors.cam_left),
            ("back", self.sensors.cam_back),
            # ("third_person", self.sensors.cam_third_person)
            ]:
            img = self.sensors.get_qimage(self.sensors.latest_images[name], cam)
            if img:
                pix = QtGui.QPixmap.fromImage(img)#.scaled(320, 240)
                self.cam_labels[name].setPixmap(pix)

        img = self.sensors.get_depthqimage(self.sensors.latest_images["rangefinder"], self.sensors.rangefinder)
        if img:
            pix = QtGui.QPixmap.fromImage(img)
            self.cam_labels["rangefinder"].setPixmap(pix)
        
        # other sensors
        gps = self.sensors.gps.getValues() if hasattr(self.sensors, "gps") else [0,0,0]
        compass = self.sensors.compass.getValues() if hasattr(self.sensors, "compass") else [0,0,0]
        with open("speed.txt", "r+") as f:
            content = f.read().strip()
            speed = float(content) if content else 0.0
        with open("acceleration.txt", "r+") as f:
            content = f.read().strip()
            acceleration = float(content) if content else 0.0

        self.speed_widget.update_speed(speed)
        self.pedals_widget.update_pedals(acceleration)
        self.speed_history.append(speed)
        self.map_widget.update_pose(gps, compass)
        self.radar_widget.update_radar()

        self.gps_label.setText(f"GPS: x={gps[0]:.2f}, y={gps[1]:.2f}, z={gps[2]:.2f}")
        self.speed_label.setText(f"Speed: {speed:.2f}")
        self.compass_label.setText(f"Compass: x={compass[0]:.2f}, y={compass[1]:.2f}, z={compass[2]:.2f}")
        self.accel_label.setText(f"Accel: {acceleration:.2f}")
   
    def eventFilter(self, obj, event):
        if obj is self.command_input and event.type() == QtCore.QEvent.KeyPress:
            # Insert key
            if event.key() == QtCore.Qt.Key_Insert:
                self.send_command()
                return True
        return super().eventFilter(obj, event)
    
    def record_audio(self):
            # Run in background thread so GUI doesn’t freeze
            threading.Thread(target=self._listen_and_process, daemon=True).start()

    def _listen_and_process(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("[INFO] Listening...")
            self.status_label.setText("🎙 Listening...")
            try:
                audio = r.listen(source, timeout=5)
                text = r.recognize_google(audio)
                print(f"[INFO] You said: {text}")
                self.status_label.setText(f"You said: {text}")
                # fill text box and trigger command
                self.command_input.setText(text)
                QtCore.QTimer.singleShot(0, self.send_command)
            except sr.WaitTimeoutError:
                print("[WARN] No speech detected.")
                self.status_label.setText("No speech detected.")
            except sr.UnknownValueError:
                print("[WARN] Could not understand audio.")
                self.status_label.setText("Could not understand audio.")
            except Exception as e:
                print("[ERROR]", e)
                self.status_label.setText("Audio error.")

class CompassWidget(QtWidgets.QWidget):
    def __init__(self, sensors, parent=None):
        super().__init__(parent)
        self.sensors = sensors

        # Figure & canvas
        self.figure = Figure(figsize=(3, 3))
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.98)
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # 2D axis (top-down view: X horizontal, Y vertical = Webots Z)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_compass)
        self.timer.start(20)  # 50 Hz; adjust as needed

    def _style_axes(self):
        self.ax.clear()
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_aspect('equal', 'box')
        self.ax.axis('off')  # hide everything (axes, labels, ticks)
        circle = plt.Circle((0, 0), 1.0, color='#cccccc', fill=False, linewidth=1.5)
        self.ax.add_patch(circle)

    def update_compass(self):
        vals = self.sensors.compass.getValues()
        wx, wy = vals[0], vals[1]
        rx, ry = wx, -wy  # heading   

        self._style_axes()

        # Fixed north arrow and label
        # self.ax.arrow(0, 0, 0, 0.25, head_width=0.06, head_length=0.06, fc='r', ec='r', linewidth=2)
        self.ax.text(0, 1.05, 'N', color='#0077cc', fontsize=8, ha='center', va='bottom', fontweight='bold', zorder=5,  bbox=dict(facecolor='none', edgecolor='none', pad=0))
        self.ax.text(0, -1.5, 'S', color='#cccccc', fontsize=8, ha='center', va='bottom', fontweight='bold', zorder=5,  bbox=dict(facecolor='none', edgecolor='none', pad=0))
        self.ax.text(-1.4, -0.25, 'W', color='#cccccc', fontsize=8, ha='center', va='bottom', fontweight='bold', zorder=5,  bbox=dict(facecolor='none', edgecolor='none', pad=0))
        self.ax.text(1.4, -0.25, 'E', color='#cccccc', fontsize=8, ha='center', va='bottom', fontweight='bold', zorder=5,  bbox=dict(facecolor='none', edgecolor='none', pad=0))



        # Robot heading arrow from origin to (rx, ry) - scale down if necessary
        scale = 0.9  # keep arrow inside circle
        mag = np.hypot(rx, ry)
        if mag > 0.001:
            sx, sy = (rx / mag) * scale, (ry / mag) * scale
        else:
            sx, sy = 0.0, 0.0

        self.ax.arrow(0, 0, sx, sy, head_width=0.06, head_length=0.06, fc="#0077cc", ec='#0077cc', linewidth=2)
        self.ax.plot([0], [0], marker='o', color='k', markersize=3)  # origin marker (vehicle position)

        # optionally show numeric heading (degrees)
        heading_deg = (np.degrees(np.arctan2(sx, sy)) + 360) % 360
        self.ax.text(1.25, 1.2, f"{heading_deg:.0f}°", transform=self.ax.transAxes,
                     fontsize=8, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#cccccc"))

        self.canvas.draw_idle()

class SpeedometerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(2, 2))
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.speed = 0
        self.max_speed = 20  # max value for gauge

    def update_speed(self, speed):
        self.speed = np.clip(speed, 0, self.max_speed)
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # geometry
        cx, cy = 0.0, -0.35   # center moved down -> arc sits lower
        r = 1.0               # radius

        # arc from left (180°) -> top (90°) -> right (0°)
        angles = np.linspace(180, 0, 200)
        xs = cx + r * np.cos(np.radians(angles))
        ys = cy + r * np.sin(np.radians(angles))
        self.ax.plot(xs, ys, color='gray', linewidth=2)

        # ticks: small every 5, big every 10; labels placed above arc (towards top)
        for s in range(0, self.max_speed + 1, 5):
            ang = 180 - (s / self.max_speed) * 180   # left->right mapping
            sx = cx + r * np.cos(np.radians(ang))
            sy = cy + r * np.sin(np.radians(ang))
            inner = 0.90 if s % 10 == 0 else 0.95
            tx = cx + inner * r * np.cos(np.radians(ang))
            ty = cy + inner * r * np.sin(np.radians(ang))
            self.ax.plot([tx, sx], [ty, sy], color='black', lw=1.5 if s % 10 == 0 else 1)

            # label for big ticks: place slightly *above* the arc (towards top of widget)
            if s % 10 == 0:
                lab_r = 1.15
                lx = cx + lab_r * r * np.cos(np.radians(ang))
                ly = cy + lab_r * r * np.sin(np.radians(ang))
                # move label a bit upward (positive y) so it's above the arc
                ly += 0.08
                self.ax.text(lx*1.05, ly, f"{s}", fontsize=7, ha='center', va='center')

        # needle: compute angle and draw from lowered center
        ang = 180 - (self.speed / self.max_speed) * 180
        nx = cx + 0.8 * r * np.cos(np.radians(ang))
        ny = cy + 0.8 * r * np.sin(np.radians(ang))
        # draw needle as a thicker arrow-like line
        self.ax.plot([cx, nx], [cy, ny], color='red', lw=2)
        # small center hub
        self.ax.plot([cx], [cy], marker='o', color='k', markersize=4)

        # speed text below center
        self.ax.text(0, cy - 0.5, f"{self.speed:.1f} km/h", ha='center', fontsize=9, fontweight='bold')

        # limits leave some padding: top should include labels above arc
        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(cy - 0.6, 1.1)
        self.canvas.draw_idle()

class SpeedHistoryWidget(QtWidgets.QWidget):
    def __init__(self, window_seconds=30, max_points=500, max_speed=20, parent=None):
        super().__init__(parent)
        self.window = float(window_seconds)
        self.max_speed = max_speed

        # data buffers
        self.times = deque()
        self.speeds = deque()

        # matplotlib setup
        self.figure = Figure(figsize=(2,1.2))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(0, self.max_speed)
        self.ax.set_xlim(-self.window, 0)
        self.ax.set_xlabel("")  # keep minimal
        self.ax.set_ylabel("") 
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.line, = self.ax.plot([], [], lw=1.5, color='#0077cc')

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.canvas)

    def append(self, speed):
        """Call this each time you have a new speed measurement (speed in same units as max_speed)."""
        now = time.time()
        self.times.append(now)
        self.speeds.append(float(speed))

        # drop old samples outside the window
        cutoff = now - self.window
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
            self.speeds.popleft()

        self._redraw(now)

    def _redraw(self, now):
        if not self.times:
            return
        xs = np.array(self.times) - now          # relative times: negative (older) -> 0 (now)
        ys = np.array(self.speeds)
        self.line.set_data(xs, ys)
        self.ax.set_xlim(-self.window, 0)

        # smart y-limits (optional): keep stable scale but allow slight autoscale
        max_seen = max(self.max_speed, ys.max() * 1.1)
        self.ax.set_ylim(0, max_seen)

        self.canvas.draw_idle()

# class MiniMapWidget(QtWidgets.QLabel):
#     def __init__(self, map_path, world_bounds, parent=None):
#         """
#         map_path: path to top-view PNG of the world.
#         world_bounds: ((min_x, max_x), (min_z, max_z)) corresponding to Webots GPS range.
#         """
#         super().__init__(parent)
#         self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
#         self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
#         self.setFixedSize(int(229*1.5), int(188*1.5))  # you can resize freely
#         self.setStyleSheet("background: transparent;")

#         self.map_pix = QtGui.QPixmap(map_path)
#         if self.map_pix.isNull():
#             raise RuntimeError("Failed to load map image with QPixmap")
#         self.world_bounds = world_bounds
#         self.car_pos = (0, 0)
#         self.heading = 0  # radians
#         # print("map path exists:", os.path.exists(map_path))
#         # print("map_pix isNull:", self.map_pix.isNull())
#         # print("map_pix size:", self.map_pix.size())

#     def update_pose(self, gps_vals, compass_vals):
#         """Update vehicle position and heading."""
#         if gps_vals is None or compass_vals is None:
#             return
#         x, y, z = gps_vals
#         self.car_pos = (x, y)

#         # test car
#         (min_x, max_x), (min_y, max_y) = self.world_bounds
#         min_x, max_x = min(min_x, max_x), max(min_x, max_x)
#         min_y, max_y = min(min_y, max_y), max(min_y, max_y)
#         world_bounds = ((min_x, max_x), (min_y, max_y))
#         inside = (min_x <= x <= max_x) and (min_y <= y <= max_y)
#         # print("world_bounds:", world_bounds, "car_pos:", self.car_pos, "inside:", inside)

#         # heading: from compass (assuming Y = north)
#         self.heading = np.arctan2(-compass_vals[0], compass_vals[1]) + np.radians(90)  # yaw angle

#         self.repaint()

#     def paintEvent(self, event):
#         painter = QtGui.QPainter(self)
#         painter.setRenderHint(QtGui.QPainter.Antialiasing)

#         if not hasattr(self, "map_pix") or self.map_pix.isNull():
#             painter.fillRect(self.rect(), QtGui.QColor("#ffffff"))
#             painter.end()
#             return

#         # draw scaled map (NO fillRect before this!)
#         scaled = self.map_pix.scaled(
#             self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
#         )

#         x_off = (self.width() - scaled.width()) // 2
#         y_off = (self.height() - scaled.height()) // 2
#         painter.drawPixmap(x_off, y_off, scaled)

#         # --- mapping ---
#         img_w, img_h = scaled.width(), scaled.height()
#         (min_x, max_x), (min_y, max_y) = self.world_bounds
#         wx, wy = self.car_pos
#         nx = (wx - min_x) / (max_x - min_x)
#         ny = (wy - min_y) / (max_y - min_y)
#         nx = max(0.0, min(1.0, nx))
#         ny = max(0.0, min(1.0, ny))
#         px = int(x_off + nx * img_w)
#         py = int(y_off + ny * img_h)

#         # --- car marker ---
#         painter.setPen(QtGui.QPen(QtGui.QColor("#95f167"), 2))
#         # painter.setBrush(QtGui.QBrush(QtCore.Qt.red)) 
#         painter.drawEllipse(px - 5, py - 5, 10, 10)
#         dx = int(np.sin(self.heading) * 12)
#         dy = int(-np.cos(self.heading) * 12)
#         painter.drawLine(px, py, px + dx, py + dy)

#         # --- border ---
#         painter.setPen(QtGui.QPen(QtCore.Qt.gray, 1))
#         painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

#         painter.end()

class MiniMapWidget(QtWidgets.QLabel):
    def __init__(self, map_path, world_bounds, parent=None, landmarks_nav=None, landmarks_xy=None):
        super().__init__(parent)
        self.setFixedSize(int(1522/5), int(1372/5))
        self.setStyleSheet("background: transparent;")
        self.setMouseTracking(True)
        self.map_pix = QtGui.QPixmap(map_path)
        if self.map_pix.isNull():
            raise RuntimeError("Failed to load map image with QPixmap")
        self.world_bounds = world_bounds  # ((min_x,max_x),(min_y,max_y))
        self.car_pos = (0.0, 0.0)
        self.heading = 0.0
        self._scaled = None
        self._x_off = 0
        self._y_off = 0

        # landmarks
        self.landmarks_nav = landmarks_nav or {}   # original dict used for navigation (may contain lane/stop strings)
        self.landmarks_xy = landmarks_xy or {}     # optional explicit mapping name -> (wx,wy)
        # try to build numeric mapping from landmarks_nav if possible
        self._landmark_points = {}
        for name, val in self.landmarks_xy.items():
            try:
                # accept ("123","-45") or (123,-45)
                wx = float(val[0]); wy = float(val[1])
                self._landmark_points[name] = (wx, wy)
            except Exception:
                pass
        # merge explicit xy (overrides)
        for name, (wx, wy) in (self.landmarks_xy.items() if self.landmarks_xy else []):
            self._landmark_points[name] = (float(wx), float(wy))

        # click marker
        self._last_click_world = None

    # -------------------
    # Mapping helpers
    # -------------------
    def _update_scaled_info(self):
        """compute scaled pixmap and offsets used for mapping"""
        if self.map_pix is None:
            return
        self._scaled = self.map_pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._x_off = (self.width() - self._scaled.width()) // 2
        self._y_off = (self.height() - self._scaled.height()) // 2
        self._img_w, self._img_h = self._scaled.width(), self._scaled.height()

    def world_to_pixel(self, wx, wy):
        """Convert world coords (wx,wy) -> pixel (px,py) on widget"""
        self._update_scaled_info()
        (min_x, max_x), (min_y, max_y) = self.world_bounds
        # normalize
        nx = (wx - min_x) / (max_x - min_x)
        ny = (wy - min_y) / (max_y - min_y)
        nx = max(0.0, min(1.0, nx)); ny = max(0.0, min(1.0, ny))
        px = int(self._x_off + nx * self._img_w)
        py = int(self._y_off + ny * self._img_h)
        return px, py

    def pixel_to_world(self, px, py):
        """Convert widget pixel coords -> world coords (wx,wy)"""
        self._update_scaled_info()
        # clamp to inside scaled image region
        ix = px - self._x_off
        iy = py - self._y_off
        ix = max(0, min(self._img_w, ix)); iy = max(0, min(self._img_h, iy))
        nx = ix / self._img_w
        ny = iy / self._img_h
        (min_x, max_x), (min_y, max_y) = self.world_bounds
        wx = min_x + nx * (max_x - min_x)
        wy = min_y + ny * (max_y - min_y)
        # print(f"wx: {wx} | wy: {wy}")
        return wx, wy

    # -------------------
    # Interaction
    # -------------------
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            px = event.pos().x(); py = event.pos().y()
            wx, wy = self.pixel_to_world(px, py)
            self._last_click_world = (wx, wy)

            # find nearest landmark (from numeric map)
            if self._landmark_points:
                name, (lx, ly), d = self._closest_landmark_to_point(wx, wy)
                # update UI and send command
                lbl = getattr(self.parent(), "status_label", None)
                if lbl is not None:
                    lbl.setText(f"Destination: {name.capitalize()}")
                # write to SUMO new_dest.txt using landmarks_nav if available (keep old behaviour)
                if name in self.landmarks_nav:
                    val = self.landmarks_nav[name]
                    try:
                        lane, stop_pos = val[0], val[1]
                        with open("new_dest.txt", "w") as f:
                            f.write(f"{lane} {stop_pos}")
                    except Exception:
                        pass
                # else: user can handle navigation by overriding parent.send_navigation_command
                if hasattr(self.parent(), "send_navigation_command"):
                    self.parent().send_navigation_command(name, (lx, ly))
            else:
                # no numeric landmarks available — try to match by name substring (best effort)
                # simply set status to clicked coords
                lbl = getattr(self.parent(), "status_label", None)
                if lbl is not None:
                    lbl.setText(f"Clicked: {wx:.1f}, {wy:.1f}")

            self.repaint()
        super().mousePressEvent(event)

    def _closest_landmark_to_point(self, wx, wy):
        """Return (name, (lx,ly), distance) of closest numeric landmark"""
        best = (None, None, float("inf"))
        for name, (lx, ly) in self._landmark_points.items():
            d = math.hypot(wx - lx, wy - ly)
            if d < best[2]:
                best = (name, (lx, ly), d)
        return best

    def mouseMoveEvent(self, event):
        pos = event.pos()
        wx, wy = self.pixel_to_world(pos.x(), pos.y())

        nearest = None
        min_dist = 15  # pixels or tune for GPS scale
        for name, (lx, ly) in self.landmarks_xy.items():
            lx, ly = float(lx), float(ly)
            px, py = self.world_to_pixel(lx, ly)
            d = ((px - pos.x())**2 + (py - pos.y())**2)**0.5
            if d < min_dist:
                nearest = name
                min_dist = d

        if nearest:
            # print(f"nearest{nearest}")
            self.setToolTip(nearest.capitalize())
        else:
            self.setToolTip("GPS Map")

    # -------------------
    # Drawing
    # -------------------
    def update_pose(self, gps_vals, compass_vals):
        if gps_vals is None:
            return
        x, y, z = gps_vals
        self.car_pos = (x, y)
        # heading same as before
        self.heading = np.arctan2(-compass_vals[0], compass_vals[1]) + np.radians(90)
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.map_pix.isNull():
            painter.fillRect(self.rect(), QtGui.QColor("#ffffff"))
            painter.end(); return

        self._update_scaled_info()
        painter.drawPixmap(self._x_off, self._y_off, self._scaled)

        # draw car
        px, py = self.world_to_pixel(*self.car_pos)
        painter.setPen(QtGui.QPen(QtGui.QColor("#95f167"), 2))
        painter.drawEllipse(px - 5, py - 5, 10, 10)
        dx = int(np.sin(self.heading) * 12); dy = int(-np.cos(self.heading) * 12)
        painter.drawLine(px, py, px + dx, py + dy)

        # draw last click (if any)
        if self._last_click_world is not None:
            cpx, cpy = self.world_to_pixel(*self._last_click_world)
            painter.setPen(QtGui.QPen(QtGui.QColor("#0077cc"), 2))
            painter.drawEllipse(cpx - 4, cpy - 4, 8, 8)

        # optionally draw landmark markers (numeric ones)
        painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        for name, (lx, ly) in self._landmark_points.items():
            lpx, lpy = self.world_to_pixel(lx, ly)
            painter.drawEllipse(lpx - 2, lpy - 2, 4, 4)

        painter.setPen(QtGui.QPen(QtCore.Qt.gray, 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        painter.end()

class PedalsWidget(QtWidgets.QWidget):
    def __init__(self, max_acc=5.0, parent=None):
        """
        max_acc: maximum absolute acceleration to normalize the bars.
        """
        super().__init__(parent)
        self.max_acc = max_acc + 0.5
        self.throttle = 0.0
        self.brake = 0.0
        self.setMinimumSize(60, 120)

    def update_pedals(self, acc):
        """
        acc > 0: throttle
        acc < 0: brake
        """
        acc = np.clip(acc, -self.max_acc, self.max_acc)
        self.throttle = max(0.0, acc) / self.max_acc
        self.brake = max(0.0, -acc) / self.max_acc
        self.update()  # triggers paintEvent

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#f0f0f0"))

        w = self.width() // 2 - 10
        h = self.height() - 15 # leave space for fixed labels

        # throttle bar
        t_height = int(self.throttle * h)
        painter.setBrush(QtGui.QColor("#2E8B57"))
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
        painter.drawRect(10, h - t_height, w, t_height)
        # numeric value
        painter.drawText(10, h - t_height - 15, w, 12,
                        QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                        f"{self.throttle*self.max_acc:.2f}")

        # brake bar
        b_height = int(self.brake * h)
        painter.setBrush(QtGui.QColor("#d9534f"))
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
        painter.drawRect(20 + w, h - b_height, w, b_height)
        painter.drawText(20 + w, h - b_height - 15, w, 12,
                        QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                        f"{self.brake*self.max_acc:.2f}")

        # fixed labels at bottom
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#2E8B57"))
        painter.drawText(10, h + 2, w, 16, QtCore.Qt.AlignHCenter, "Throttle")
        painter.setPen(QtGui.QColor("#d9534f"))
        painter.drawText(20 + w, h + 2, w, 16, QtCore.Qt.AlignHCenter, "Brake")

        painter.end()

class RadarWidget(QtWidgets.QWidget):
    def __init__(self, radar_front, radar_back, max_range=None, parent=None):
        super().__init__(parent)
        self.radar_front = radar_front
        self.radar_back = radar_back
        # try to get max range from device if available, otherwise use fallback
        if max_range is None and hasattr(radar_front, "getMaxRange"):
            try:
                max_range = float(radar_front.getMaxRange())
            except Exception:
                max_range = 50.0
        self.max_range = float(max_range or 50.0)
        self.data = []  # list of tuples (dist, azimuth, power, speed)
        self.setMinimumSize(120, 120)
        self.setMaximumSize(120, 120)

    def update_radar(self):
        """Read radar targets and store (distance, azimuth, power, speed)."""
        if self.radar_front is None or self.radar_back is None:
            return
        n_front = self.radar_front.getNumberOfTargets()
        n_back = self.radar_back.getNumberOfTargets()
        if n_front == 0 and n_back == 0:
            self.data = []
            self.repaint()
            return

        targets_front = self.radar_front.getTargets()
        targets_back = self.radar_back.getTargets()
        targets = targets_front + targets_back
        out = []
        for t in targets:
            try:
                dist = float(getattr(t, "distance", 0.0))
                az   = float(getattr(t, "azimuth", 0.0))    # expected radians
                if t in targets_back:
                    az += np.pi  # rotate 180° for rear radar
                power= float(getattr(t, "receiver_power", 0.0))
                spd  = float(getattr(t, "speed", 0.0))
                out.append((dist, az, power, spd))
            except Exception:
                # skip malformed target
                continue
        self.data = out
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # background (choose your color or remove to keep transparent)
        painter.fillRect(self.rect(), QtGui.QColor("#202020"))

        if not self.data:
            # draw empty radar circle
            w, h = self.width(), self.height()
            cx, cy = w//2, h//2
            radius = min(cx, cy) - 8
            painter.setPen(QtGui.QPen(QtCore.Qt.gray, 1))
            painter.drawEllipse(cx - radius, cy - radius, 2*radius, 2*radius)
            painter.end()
            return

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(cx, cy) - 8

        # draw concentric range rings
        painter.setPen(QtGui.QPen(QtCore.Qt.darkGray, 1, QtCore.Qt.DotLine))
        for frac in (0.25, 0.5, 0.75, 1.0):
            r = int(radius * frac)
            painter.drawEllipse(cx - r, cy - r, 2*r, 2*r)

        # draw targets
        for dist, az, power, spd in self.data:
            if dist <= 0 or dist > self.max_range:
                continue
            # map distance -> pixel radius
            r_px = (dist / self.max_range) * radius
            x = cx + r_px * np.cos(az-np.radians(90))
            y = cy - r_px * -np.sin(az-np.radians(90))

            # color intensity from receiver_power (normalize heuristically)
            # clamp power to sensible range for alpha (you may tune min/max)
            alpha = np.clip((power + 80) / 80, 0.05, 1.0)  # example mapping
            col = QtGui.QColor(0, 200, 0)
            col.setAlphaF(float(alpha))

            # size by speed (small base + scaled)
            size = max(3, min(12, 3 + abs(spd) * 0.5))

            painter.setBrush(QtGui.QBrush(col))
            painter.setPen(QtGui.QPen(col.darker(), 1))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))

        # outline
        painter.setPen(QtGui.QPen(QtCore.Qt.gray, 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(cx - radius, cy - radius, 2*radius, 2*radius)

        painter.end()



# ----------------------------
#  Main program
# ----------------------------
if __name__ == "__main__":
    try:
        sensors = WebotsSensors()
        sensors.start()

        app = QtWidgets.QApplication(sys.argv)
        dashboard = Dashboard(sensors)
        dashboard.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("[ERROR] main gui crashed:", e)
        traceback.print_exc()
