from flask import Flask, render_template, request, redirect, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import math
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "12345"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///parking.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

model = None
best_threshold = 0.5
model_path = os.path.join(os.path.dirname(__file__), "availability_model.pkl")

if os.path.exists(model_path):
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        model = bundle.get("model")
        best_threshold = bundle.get("threshold", 0.5)
    else:
        model = bundle

model_accuracy = None

def calculate_model_accuracy():
    global model_accuracy

    try:
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "smart_parking_improved.csv"
        )

        if model is None:
            print("Accuracy skipped: model not loaded")
            model_accuracy = None
            return

        if not os.path.exists(dataset_path):
            print("Accuracy skipped: dataset not found")
            model_accuracy = None
            return

        df = pd.read_csv(dataset_path)

        # ---------------------------
        # CLEAN DATA
        # ---------------------------
        df = df.drop_duplicates().copy()
        df["occupied"] = pd.to_numeric(df["occupied"], errors="coerce")
        df = df.dropna(subset=["slot_id", "entry_hour", "exit_hour", "day_of_week", "is_weekend", "occupied"])

        df["slot_id"] = df["slot_id"].astype(int)
        df["entry_hour"] = df["entry_hour"].astype(int).clip(0, 23)
        df["exit_hour"] = df["exit_hour"].astype(int).clip(0, 23)
        df["day_of_week"] = df["day_of_week"].astype(int).clip(0, 6)
        df["is_weekend"] = df["is_weekend"].astype(int)
        df["occupied"] = df["occupied"].astype(int)

        # ---------------------------
        # FEATURE ENGINEERING
        # ---------------------------
        df["duration"] = (df["exit_hour"] - df["entry_hour"]) % 24
        df.loc[df["duration"] <= 0, "duration"] = 1

        df["entry_sin"] = np.sin(2 * np.pi * df["entry_hour"] / 24)
        df["entry_cos"] = np.cos(2 * np.pi * df["entry_hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["peak_hour"] = df["entry_hour"].apply(lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 20) else 0)
        df["is_night"] = df["entry_hour"].apply(lambda x: 1 if (x >= 22 or x <= 5) else 0)
        df["time_block"] = df["entry_hour"] // 3

        df["slot_group"] = df["slot_id"] % 5
        df["slot_zone"] = df["slot_id"] % 3

        df["traffic_level"] = df["entry_hour"].apply(
            lambda x: 2 if (7 <= x <= 10 or 17 <= x <= 20)
            else 1 if (10 < x < 17)
            else 0
        )

        df["location_type"] = df["slot_id"].apply(lambda x: 1 if x % 3 == 0 else 0)
        df["is_long_stay"] = df["duration"].apply(lambda x: 1 if x >= 5 else 0)
        df["is_short_stay"] = df["duration"].apply(lambda x: 1 if x <= 2 else 0)
        df["rush_factor"] = df["peak_hour"] * (df["is_weekend"] + 1)
        df["work_rush"] = (1 - df["is_weekend"]) * df["traffic_level"]
        df["weekend_peak"] = df["is_weekend"] * df["peak_hour"]
        df["hour_slot_interaction"] = df["entry_hour"] * df["slot_group"]

        # ---------------------------
        # FEATURES
        # ---------------------------
        feature_cols = [
            "slot_id",
            "entry_hour",
            "exit_hour",
            "day_of_week",
            "is_weekend",
            "duration",
            "entry_sin",
            "entry_cos",
            "day_sin",
            "day_cos",
            "peak_hour",
            "is_night",
            "time_block",
            "slot_group",
            "slot_zone",
            "traffic_level",
            "location_type",
            "is_long_stay",
            "is_short_stay",
            "rush_factor",
            "work_rush",
            "weekend_peak",
            "hour_slot_interaction"
        ]

        X = df[feature_cols]
        y = df["occupied"]

        # model bundle or plain model support
        current_model = model["model"] if isinstance(model, dict) else model

        y_pred = current_model.predict(X)
        model_accuracy = round(accuracy_score(y, y_pred) * 100, 2)

        print(" Model Accuracy:", model_accuracy, "%")

    except Exception as e:
        print("Accuracy error:", e)
        model_accuracy = None

PLACE_COORDINATES = {
    "PVP Mall, Vijayawada, India": (16.5008, 80.6415),
    "Bus Stand, Vijayawada, India": (16.5090, 80.6460),
    "Benz Circle, Vijayawada, India": (16.5166, 80.6220),
    "Kanaka Durga Temple, Vijayawada, India": (16.5142, 80.6185),
    "Vijayawada Railway Station, Vijayawada, India": (16.5148, 80.6423),
    "MG Road, Vijayawada, India": (16.5120, 80.6400),
    "Governorpet, Vijayawada, India": (16.5110, 80.6480),
    "Auto Nagar, Vijayawada, India": (16.5230, 80.6290),
    "City Centre, Vijayawada, India": (16.5155, 80.6320),
    "Patamata, Vijayawada, India": (16.5205, 80.6360),
    "Tadepalli, Vijayawada, India": (16.4800, 80.6210),
    "Bhavani Island": (16.4833, 80.5667),
    "Prakasam Barrage": (16.5167, 80.6167),
    "Gandhi Hill": (16.5165, 80.6185),
    "Undavalli Caves": (16.4839, 80.6078),
    "Gannavaram Airport": (16.5304, 80.7977),
    "PVP Square Mall": (16.5065, 80.6482),
}
PRICE_PER_KM = {"2-wheeler": 5, "3-wheeler": 8, "4-wheeler": 12}

class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    wallet_balance = db.Column(db.Float, default=0)

class WalletTransaction(db.Model):
    __tablename__ = "wallet_transaction"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    tx_type = db.Column(db.String(50))
    reference_type = db.Column(db.String(50))
    reference_id = db.Column(db.Integer)
    location = db.Column(db.String(200))
    amount = db.Column(db.Float, default=0)
    created_at = db.Column(db.String(50))

class ParkingArea(db.Model):
    __tablename__ = "parking_area"
    id = db.Column(db.Integer, primary_key=True)
    location_name = db.Column(db.String(200), unique=True, nullable=False)

class ParkingSlot(db.Model):
    __tablename__ = "parking_slot"
    id = db.Column(db.Integer, primary_key=True)
    slot_number = db.Column(db.Integer, nullable=False)
    area_id = db.Column(db.Integer, db.ForeignKey("parking_area.id"), nullable=False)

class ParkingBooking(db.Model):
    __tablename__ = "parking_booking"
    id = db.Column(db.Integer, primary_key=True)
    booking_code = db.Column(db.String(100), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    slot_id = db.Column(db.Integer, db.ForeignKey("parking_slot.id"), nullable=False)
    entry_time = db.Column(db.String(50), nullable=False)
    exit_time = db.Column(db.String(50), nullable=False)
    payment = db.Column(db.String(50), default="upi")
    amount = db.Column(db.Float, default=0)
    transaction_id = db.Column(db.String(100), unique=True)
    payment_status = db.Column(db.String(50), default="verified")
    status = db.Column(db.String(50), default="confirmed")
    refund_amount = db.Column(db.Float, default=0)
    cancelled_at = db.Column(db.String(50))
    released_to_instant = db.Column(db.Boolean, default=False)
    instant_released_at = db.Column(db.String(50))
    booked_at = db.Column(db.String(50))
    rate_per_hour = db.Column(db.Float, default=20)
    is_peak_hour = db.Column(db.Boolean, default=False)
    is_weekend = db.Column(db.Boolean, default=False)
    is_instant = db.Column(db.Boolean, default=False)

class EVStation(db.Model):
    __tablename__ = "ev_station"
    id = db.Column(db.Integer, primary_key=True)
    location_name = db.Column(db.String(200), unique=True, nullable=False)


class EVSlot(db.Model):
    __tablename__ = "ev_slot"
    id = db.Column(db.Integer, primary_key=True)
    slot_number = db.Column(db.Integer, nullable=False)
    station_id = db.Column(db.Integer, db.ForeignKey("ev_station.id"), nullable=False)
    charger_type = db.Column(db.String(50), default="Fast")
    connector_type = db.Column(db.String(50), default="CCS")
    power = db.Column(db.String(50), default="50kW")


class EVBooking(db.Model):
    __tablename__ = "ev_booking"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    slot_id = db.Column(db.Integer, db.ForeignKey("ev_slot.id"), nullable=False)
    entry_time = db.Column(db.String(50), nullable=False)
    exit_time = db.Column(db.String(50), nullable=False)
    connector_type = db.Column(db.String(50))
    amount = db.Column(db.Float, default=0)
    transaction_id = db.Column(db.String(100), unique=True)
    payment_status = db.Column(db.String(50), default="verified")
    status = db.Column(db.String(50), default="confirmed")
    refund_amount = db.Column(db.Float, default=0)
    cancelled_at = db.Column(db.String(50))
    released_to_instant = db.Column(db.Boolean, default=False)
    instant_released_at = db.Column(db.String(50))
    booked_at = db.Column(db.String(50))

class CarpoolRide(db.Model):
    __tablename__ = "carpool_ride"
    id = db.Column(db.Integer, primary_key=True)
    driver_name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    vehicle_type = db.Column(db.String(50))
    vehicle = db.Column(db.String(100))
    vehicle_number = db.Column(db.String(50))
    start_location = db.Column(db.String(200))
    destination = db.Column(db.String(200))
    date_time = db.Column(db.String(50))
    available_seats = db.Column(db.Integer, default=0)
    price = db.Column(db.Float, default=0)
    notes = db.Column(db.String(300))
    is_shared = db.Column(db.Boolean, default=True)
    driver_lat = db.Column(db.Float)
    driver_lng = db.Column(db.Float)
    driver_dest_lat = db.Column(db.Float)
    driver_dest_lng = db.Column(db.Float)

class CarpoolBooking(db.Model):
    __tablename__ = "carpool_booking"
    id = db.Column(db.Integer, primary_key=True)
    ride_id = db.Column(db.Integer, db.ForeignKey("carpool_ride.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    seats_booked = db.Column(db.Integer, default=1)
    booked_at = db.Column(db.String(50))

def normalize_location_name(name):
    return str(name).lower().replace("square", "").replace(",", "").replace("india", "").strip()

def is_float_value(value):
    try:
        float(value)
        return True
    except:
        return False

def calculate_distance(lat1, lon1, lat2, lon2):
    r = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c

def get_nearest_area_from_coords(lat, lng):
    nearest_area = None
    min_distance = float("inf")

    MAX_DISTANCE_KM = 2.5   # 🔥 important (change if needed)

    for area in ParkingArea.query.all():
        area_name = area.location_name
        area_lat, area_lng = PLACE_COORDINATES.get(area_name, (None, None))

        if area_lat is None or area_lng is None:
            continue

        dist = calculate_distance(lat, lng, area_lat, area_lng)

        if dist < min_distance:
            min_distance = dist
            nearest_area = area

    # 🔥 NEW LOGIC
    if nearest_area and min_distance <= MAX_DISTANCE_KM:
        return nearest_area
    else:
        return None

def get_lat_lng(place):
    return PLACE_COORDINATES.get(place, (None, None))

def calculate_parking_refund(payment_value):
    if payment_value is None:
        return 0
    try:
        return round(float(str(payment_value).replace("₹", "").strip()) / 3, 2)
    except Exception:
        return 0

def get_dynamic_price(entry_dt, exit_dt, is_instant=False):
    duration_hours = (exit_dt - entry_dt).total_seconds() / 3600
    if duration_hours <= 0:
        duration_hours = 1
    base_rate = 20
    entry_hour = entry_dt.hour
    is_weekend = entry_dt.weekday() >= 5
    is_peak_hour = (7 <= entry_hour <= 10) or (17 <= entry_hour <= 21)
    rate = base_rate + (10 if is_peak_hour else 0) + (5 if is_weekend else 0) + (10 if is_instant else 0)
    return {"rate_per_hour": rate, "duration_hours": round(duration_hours, 2), "total_amount": round(duration_hours * rate, 2), "is_peak_hour": is_peak_hour, "is_weekend": is_weekend, "is_instant": is_instant}

def is_duplicate_transaction_id(transaction_id):
    if not transaction_id:
        return False
    return (ParkingBooking.query.filter_by(transaction_id=transaction_id).first() is not None or EVBooking.query.filter_by(transaction_id=transaction_id).first() is not None)

def predict_future_availability(slot_id, entry_dt, exit_dt):
    if model is None:
        return {
            "predicted_occupied": None,
            "probability_available": None,
            "probability_occupied": None
        }

    try:
        entry_hour = entry_dt.hour
        exit_hour = exit_dt.hour
        day_of_week = entry_dt.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        duration = (exit_hour - entry_hour) % 24
        if duration <= 0:
            duration = 1

        entry_sin = math.sin(2 * math.pi * entry_hour / 24)
        entry_cos = math.cos(2 * math.pi * entry_hour / 24)
        day_sin = math.sin(2 * math.pi * day_of_week / 7)
        day_cos = math.cos(2 * math.pi * day_of_week / 7)

        peak_hour = 1 if (7 <= entry_hour <= 10 or 17 <= entry_hour <= 20) else 0
        is_night = 1 if (entry_hour >= 22 or entry_hour <= 5) else 0
        time_block = entry_hour // 3
        slot_group = slot_id % 5
        slot_zone = slot_id % 3

        traffic_level = 2 if (7 <= entry_hour <= 10 or 17 <= entry_hour <= 20) else (1 if (10 < entry_hour < 17) else 0)
        location_type = 1 if slot_id % 3 == 0 else 0
        is_long_stay = 1 if duration >= 5 else 0
        is_short_stay = 1 if duration <= 2 else 0
        rush_factor = peak_hour * (is_weekend + 1)
        work_rush = (1 - is_weekend) * traffic_level
        weekend_peak = is_weekend * peak_hour
        hour_slot_interaction = entry_hour * slot_group

        features = pd.DataFrame([{
            "slot_id": int(slot_id),
            "entry_hour": int(entry_hour),
            "exit_hour": int(exit_hour),
            "day_of_week": int(day_of_week),
            "is_weekend": int(is_weekend),
            "duration": float(duration),
            "entry_sin": float(entry_sin),
            "entry_cos": float(entry_cos),
            "day_sin": float(day_sin),
            "day_cos": float(day_cos),
            "peak_hour": int(peak_hour),
            "is_night": int(is_night),
            "time_block": int(time_block),
            "slot_group": int(slot_group),
            "slot_zone": int(slot_zone),
            "traffic_level": int(traffic_level),
            "location_type": int(location_type),
            "is_long_stay": int(is_long_stay),
            "is_short_stay": int(is_short_stay),
            "rush_factor": int(rush_factor),
            "work_rush": int(work_rush),
            "weekend_peak": int(weekend_peak),
            "hour_slot_interaction": int(hour_slot_interaction)
        }])

        prob = model.predict_proba(features)[0]
        pred = 1 if prob[1] > best_threshold else 0

        return {
            "predicted_occupied": int(pred),
            "probability_available": float(prob[0]),
            "probability_occupied": float(prob[1])
        }

    except Exception as e:
        print("Prediction error:", e)
        return {
            "predicted_occupied": None,
            "probability_available": None,
            "probability_occupied": None
        }

def get_nearby_available_areas(selected_location, req_entry, req_exit, limit=3):
    selected_lat, selected_lng = PLACE_COORDINATES.get(selected_location, (None, None))
    if selected_lat is None:
        return []
    selected_norm = normalize_location_name(selected_location)
    nearby = []
    seen = set()
    for area in ParkingArea.query.all():
        area_name = area.location_name
        area_norm = normalize_location_name(area_name)
        if area_norm == selected_norm:
            continue
        area_lat, area_lng = PLACE_COORDINATES.get(area_name, (None, None))
        if area_lat is None:
            continue
        has_free = False
        for slot in ParkingSlot.query.filter_by(area_id=area.id).all():
            overlap = ParkingBooking.query.filter(ParkingBooking.slot_id == slot.id, ParkingBooking.status == "confirmed", ParkingBooking.entry_time < req_exit.isoformat(timespec="minutes"), ParkingBooking.exit_time > req_entry.isoformat(timespec="minutes")).first()
            if not overlap:
                has_free = True
                break
        if has_free and area_norm not in seen:
            seen.add(area_norm)
            nearby.append({"location_name": area_name, "distance_km": round(calculate_distance(selected_lat, selected_lng, area_lat, area_lng), 2)})
    nearby.sort(key=lambda x: x["distance_km"])
    return nearby[:limit]

def seed_parking_areas():
    locations = list(PLACE_COORDINATES.keys())
    next_slot_id = 1
    last_slot = ParkingSlot.query.order_by(ParkingSlot.id.desc()).first()
    if last_slot:
        next_slot_id = last_slot.id + 1
    for location in locations:
        area = ParkingArea.query.filter_by(location_name=location).first()
        if not area:
            area = ParkingArea(location_name=location)
            db.session.add(area)
            db.session.commit()
        if ParkingSlot.query.filter_by(area_id=area.id).count() == 0:
            for slot_number in range(1, 21):
                db.session.add(ParkingSlot(id=next_slot_id, slot_number=slot_number, area_id=area.id))
                next_slot_id += 1
            db.session.commit()

def seed_ev_stations():
    stations = ["PVP Mall, Vijayawada, India", "Benz Circle, Vijayawada, India", "Gannavaram Airport", "Bhavani Island", "MG Road, Vijayawada, India"]
    next_slot_id = 1
    last_slot = EVSlot.query.order_by(EVSlot.id.desc()).first()
    if last_slot:
        next_slot_id = last_slot.id + 1
    for station_name in stations:
        station = EVStation.query.filter_by(location_name=station_name).first()
        if not station:
            station = EVStation(location_name=station_name)
            db.session.add(station)
            db.session.commit()
        if EVSlot.query.filter_by(station_id=station.id).count() == 0:
            templates = [("Fast", "CCS", "50kW"), ("Fast", "CCS", "50kW"), ("Slow", "Type2", "11kW"), ("Slow", "Type2", "11kW")]
            for idx, tpl in enumerate(templates, start=1):
                db.session.add(EVSlot(id=next_slot_id, slot_number=idx, station_id=station.id, charger_type=tpl[0], connector_type=tpl[1], power=tpl[2]))
                next_slot_id += 1
            db.session.commit()

def seed_demo_carpool():
    if CarpoolRide.query.count() > 0:
        return
    rides = [
        {"driver_name": "Ravi", "phone": "9876543210", "vehicle_type": "4-wheeler", "vehicle": "Swift Dzire", "vehicle_number": "AP16AB1234", "start_location": "PVP Mall, Vijayawada, India", "destination": "Benz Circle, Vijayawada, India", "date_time": datetime.now().isoformat(timespec="minutes"), "available_seats": 3, "price": 80, "notes": "Morning office ride", "is_shared": True},
        {"driver_name": "Suresh", "phone": "9123456780", "vehicle_type": "2-wheeler", "vehicle": "Activa", "vehicle_number": "AP16CD5678", "start_location": "Bhavani Island", "destination": "MG Road, Vijayawada, India", "date_time": datetime.now().isoformat(timespec="minutes"), "available_seats": 1, "price": 40, "notes": "Evening ride", "is_shared": True}
    ]
    for r in rides:
        s_lat, s_lng = get_lat_lng(r["start_location"])
        d_lat, d_lng = get_lat_lng(r["destination"])
        db.session.add(CarpoolRide(driver_name=r["driver_name"], phone=r["phone"], vehicle_type=r["vehicle_type"], vehicle=r["vehicle"], vehicle_number=r["vehicle_number"], start_location=r["start_location"], destination=r["destination"], date_time=r["date_time"], available_seats=r["available_seats"], price=r["price"], notes=r["notes"], is_shared=r["is_shared"], driver_lat=s_lat, driver_lng=s_lng, driver_dest_lat=d_lat, driver_dest_lng=d_lng))
    db.session.commit()

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            return "User already exists!"
        db.session.add(User(username=username, password=password, wallet_balance=0))
        db.session.commit()
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session["user"] = user.username
            return redirect("/")
        return "Invalid credentials!"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

@app.route("/")
def index():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html", user=session["user"], model_accuracy=model_accuracy)

@app.route("/slots", methods=["GET", "POST"])
def slots():
    if "user" not in session:
        return redirect("/login")

    areas = ParkingArea.query.order_by(ParkingArea.location_name).all()

    location = request.args.get("location") or request.form.get("location")
    entry_time = request.args.get("entry_time", "")
    exit_time = request.args.get("exit_time", "")

    filtered_areas = [a for a in areas if a.location_name == location] if location else areas

    return render_template(
        "slots.html",
        areas=filtered_areas,
        user=session["user"],
        model_accuracy=model_accuracy,
        selected_location=location or "",
        selected_entry_time=entry_time,
        selected_exit_time=exit_time
    )

@app.route("/book/<int:slot_id>", methods=["GET", "POST"])
def book(slot_id):
    if "user" not in session:
        return redirect("/login")

    slot = ParkingSlot.query.get(slot_id)
    if not slot:
        return "❌ Slot not found"

    area = ParkingArea.query.get(slot.area_id)

    entry_time = request.args.get("entry_time") or request.form.get("entry_time")
    exit_time = request.args.get("exit_time") or request.form.get("exit_time")

    amount = 0
    pricing = None

    if entry_time and exit_time:
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            exit_dt = datetime.fromisoformat(exit_time)

            is_instant = ParkingBooking.query.filter_by(
                slot_id=slot.id,
                status="cancelled",
                released_to_instant=True
            ).first() is not None

            pricing = get_dynamic_price(entry_dt, exit_dt, is_instant=is_instant)
            amount = pricing["total_amount"]
        except Exception:
            pass

    if request.method == "POST":
        transaction_id = request.form.get("transaction_id", "").strip()

        if not transaction_id:
            return "❌ Please enter UPI Transaction ID / UTR Number"

        if len(transaction_id) < 8:
            return "❌ Please enter a valid UPI Transaction ID / UTR Number"

        if is_duplicate_transaction_id(transaction_id):
            return "❌ This Transaction ID is already used. Please enter a valid one."

        entry_value = request.form.get("entry_time")
        exit_value = request.form.get("exit_time")

        try:
            entry_dt = datetime.fromisoformat(entry_value)
            exit_dt = datetime.fromisoformat(exit_value)
        except Exception:
            return "❌ Invalid date/time format"

        now = datetime.now()

        if entry_dt < now or exit_dt < now:
            return "❌ Previous dates and times are not allowed"

        if exit_dt <= entry_dt:
            return "❌ Exit time must be after entry time"

        # overlap check
        existing_bookings = ParkingBooking.query.filter_by(
            slot_id=slot.id,
            status="confirmed"
        ).all()

        for existing in existing_bookings:
            existing_entry = datetime.fromisoformat(existing.entry_time)
            existing_exit = datetime.fromisoformat(existing.exit_time)

            if entry_dt < existing_exit and exit_dt > existing_entry:
                return f"❌ Slot already booked during this time. Free after {existing_exit.strftime('%Y-%m-%d %H:%M')}"

        user = User.query.filter_by(username=session["user"]).first()

        booking = ParkingBooking(
            booking_code=f"PK-{slot.id}-{int(datetime.now().timestamp())}",
            user_id=user.id,
            slot_id=slot.id,
            entry_time=entry_value,
            exit_time=exit_value,
            payment="upi",
            amount=float(request.form.get("amount", 0) or 0),
            transaction_id=transaction_id,
            payment_status="verified",
            status="confirmed",
            refund_amount=0,
            booked_at=datetime.now().isoformat(timespec="minutes"),
            rate_per_hour=pricing["rate_per_hour"] if pricing else 20,
            is_peak_hour=pricing["is_peak_hour"] if pricing else False,
            is_weekend=pricing["is_weekend"] if pricing else False,
            is_instant=pricing["is_instant"] if pricing else False
        )

        db.session.add(booking)
        db.session.commit()

        return redirect("/my_bookings")

    return render_template(
        "booking.html",
        slot=slot,
        location=area.location_name,
        entry_time=entry_time,
        exit_time=exit_time,
        amount=amount,
        pricing=pricing
    )

@app.route("/cancel/<int:slot_id>", methods=["POST"])
def cancel_parking_booking(slot_id):
    if "user" not in session:
        return redirect("/login")
    user = User.query.filter_by(username=session["user"]).first()
    booking = ParkingBooking.query.filter_by(slot_id=slot_id, user_id=user.id, status="confirmed").order_by(ParkingBooking.id.desc()).first()
    if not booking:
        return "❌ Active booking not found for this slot"
    refund_given = calculate_parking_refund(booking.amount)
    booking.status = "cancelled"
    booking.refund_amount = refund_given
    booking.cancelled_at = datetime.now().isoformat(timespec="minutes")
    booking.released_to_instant = True
    booking.instant_released_at = datetime.now().isoformat(timespec="minutes")
    user.wallet_balance += refund_given
    slot = ParkingSlot.query.get(slot_id)
    area = ParkingArea.query.get(slot.area_id)
    db.session.add(WalletTransaction(user_id=user.id, tx_type="Parking Refund", reference_type="parking", reference_id=booking.id, location=area.location_name, amount=refund_given, created_at=datetime.now().isoformat(timespec="minutes")))
    db.session.commit()
    return redirect("/my_bookings")

@app.route("/get_available_slots")
def get_available_slots():
    try:
        if "user" not in session:
            return jsonify({"error": "not logged in"}), 403

        location = request.args.get("location", "").strip()
        entry_time = request.args.get("entry_time", "").strip()
        exit_time = request.args.get("exit_time", "").strip()

        if not location or not entry_time or not exit_time:
            return jsonify({"error": "location, entry_time and exit_time are required"}), 400

        req_entry = datetime.fromisoformat(entry_time)
        req_exit = datetime.fromisoformat(exit_time)

        now = datetime.now()
        if req_entry < now or req_exit < now:
            return jsonify({"error": "Previous dates and times are not allowed"}), 400

        if req_exit <= req_entry:
            return jsonify({"error": "Exit time must be after entry time"}), 400

        area = None

        # ============================
        # 1) COORDINATE SEARCH
        # ============================
        parts = [x.strip() for x in location.split(",")]

        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lng = float(parts[1])

                nearest_area = None
                min_distance = float("inf")
                MAX_DISTANCE_KM = 2.5

                for area_obj in ParkingArea.query.all():
                    area_name = area_obj.location_name
                    area_lat, area_lng = PLACE_COORDINATES.get(area_name, (None, None))

                    if area_lat is None or area_lng is None:
                        continue

                    dist = calculate_distance(lat, lng, area_lat, area_lng)

                    if dist < min_distance:
                        min_distance = dist
                        nearest_area = area_obj

                if nearest_area and min_distance <= MAX_DISTANCE_KM:
                    area = nearest_area
                elif nearest_area and min_distance > MAX_DISTANCE_KM:
                    return jsonify({"error": "No parking available near selected location"}), 404

            except ValueError:
                area = None

        # ============================
        # 2) STRONG NAME SEARCH
        # ============================
        if not area:
            search = location.strip().lower()

            # contains-based search
            for area_obj in ParkingArea.query.all():
                db_name = area_obj.location_name.lower()
                if search in db_name:
                    area = area_obj
                    break

            # normalized exact fallback
            if not area:
                normalized_input = normalize_location_name(location)

                for area_obj in ParkingArea.query.all():
                    if normalize_location_name(area_obj.location_name) == normalized_input:
                        area = area_obj
                        break

        if not area:
            return jsonify({"error": f"Location '{location}' not found"}), 404

        # ============================
        # 3) SLOT LOGIC
        # ============================
        results = []
        slots = ParkingSlot.query.filter_by(area_id=area.id).order_by(ParkingSlot.slot_number).all()

        for index, slot in enumerate(slots, start=1):
            status = "Free"
            free_at = None

            confirmed_bookings = ParkingBooking.query.filter_by(
                slot_id=slot.id,
                status="confirmed"
            ).all()

            for booking in confirmed_bookings:
                try:
                    booked_entry = datetime.fromisoformat(booking.entry_time)
                    booked_exit = datetime.fromisoformat(booking.exit_time)

                    # overlap check
                    if req_entry < booked_exit and req_exit > booked_entry:
                        status = "Booked"
                        free_at = booked_exit.isoformat(timespec="minutes")
                        break
                except Exception:
                    continue

            prediction = predict_future_availability(slot.id, req_entry, req_exit)

            latest_cancelled = ParkingBooking.query.filter_by(
                slot_id=slot.id,
                status="cancelled",
                released_to_instant=True
            ).order_by(ParkingBooking.id.desc()).first()

            results.append({
                "id": slot.id,
                "slot_number": slot.slot_number or index,
                "status": status,
                "free_at": free_at,
                "is_instant": status == "Free" and latest_cancelled is not None,
                "instant_released_at": latest_cancelled.instant_released_at if latest_cancelled else None,
                "predicted_occupied": prediction["predicted_occupied"],
                "probability_available": prediction["probability_available"],
                "probability_occupied": prediction["probability_occupied"]
            })

        return jsonify({
            "areas": [{
                "location_name": area.location_name,
                "slots": results
            }],
            "nearby_suggestions": get_nearby_available_areas(
                area.location_name,
                req_entry,
                req_exit
            )
        })

    except Exception as e:
        print("GET_AVAILABLE_SLOTS ERROR:", str(e))
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/ev/stations")
def ev_stations():
    if "user" not in session:
        return redirect("/login")

    db_stations = EVStation.query.order_by(EVStation.location_name).all()
    stations_data = []

    for s in db_stations:
        lat, lng = get_lat_lng(s.location_name)
        stations_data.append({
            "id": s.id,
            "location_name": s.location_name,
            "name": s.location_name,
            "latitude": lat,
            "longitude": lng
        })

    return render_template("ev_map.html", stations=stations_data)
@app.route("/ev/slots")
def ev_slots_json():

    # 🔒 Login check
    if "user" not in session:
        return {"error": "not logged in"}, 403

    # 📥 Get inputs
    location = request.args.get("location", "").strip()
    entry_time = request.args.get("entry_time", "").strip()
    exit_time = request.args.get("exit_time", "").strip()

    print("SEARCH LOCATION:", location)

    if not location or not entry_time or not exit_time:
        return {"error": "location, entry_time and exit_time are required"}, 400

    # 🔍 Find station (exact match)
    station = EVStation.query.filter(
        EVStation.location_name.ilike(location)
    ).first()

    # 🔁 Fallback match (normalized)
    if not station:
        normalized_input = normalize_location_name(location)
        for s in EVStation.query.all():
            if normalize_location_name(s.location_name) == normalized_input:
                station = s
                break

    print("MATCHED STATION:", station.location_name if station else "NONE")

    if not station:
        return {"error": f"EV station not found for '{location}'"}, 404

    # 🕒 Convert time
    try:
        dt_entry = datetime.fromisoformat(entry_time)
        dt_exit = datetime.fromisoformat(exit_time)
    except Exception:
        return {"error": "invalid date format"}, 400

    if dt_exit <= dt_entry:
        return {"error": "exit time must be after entry time"}, 400

    # 📊 Check slots count
    slot_count = EVSlot.query.filter_by(station_id=station.id).count()
    print("EV SLOT COUNT:", slot_count)

    slots_info = []

    # 🔄 Loop through slots
    for s in EVSlot.query.filter_by(station_id=station.id).order_by(EVSlot.slot_number).all():

        # ⚠️ FIX: use datetime comparison (NOT string)
        overlapping = EVBooking.query.filter(
            EVBooking.slot_id == s.id,
            EVBooking.status == "confirmed",
            EVBooking.entry_time < dt_exit,
            EVBooking.exit_time > dt_entry
        ).first()

        slots_info.append({
            "id": s.id,
            "slot_number": s.slot_number,
            "is_available": overlapping is None,
            "next_available_time": overlapping.exit_time.strftime("%Y-%m-%d %H:%M") if overlapping else "Now",
            "charger_type": s.charger_type,
            "connector_type": s.connector_type,
            "power": s.power
        })

    return {"slots": slots_info}

    
@app.route("/ev/book/<int:slot_id>", methods=["GET", "POST"])
def ev_book(slot_id):
    if "user" not in session:
        return redirect("/login")
    slot = EVSlot.query.get(slot_id)
    if not slot:
        return "❌ Slot not found!", 404
    station = EVStation.query.get(slot.station_id)
    if request.method == "POST":
        start_time = request.form.get("start_time")
        end_time = request.form.get("end_time")
        connector_type = request.form.get("connector_type") or slot.connector_type
        transaction_id = request.form.get("transaction_id", "").strip()
        if transaction_id and is_duplicate_transaction_id(transaction_id):
            return "❌ This Transaction ID is already used. Please enter a valid one."
        dt_start = datetime.fromisoformat(start_time)
        dt_end = datetime.fromisoformat(end_time)
        now = datetime.now()
        if dt_start < now or dt_end < now:
            return "❌ Previous dates and times are not allowed", 400
        if dt_end <= dt_start:
            return "❌ End time must be after start time", 400
        overlap = EVBooking.query.filter(EVBooking.slot_id == slot.id, EVBooking.status == "confirmed", EVBooking.entry_time < end_time, EVBooking.exit_time > start_time).first()
        if overlap:
            return f"❌ Slot already booked until {overlap.exit_time}", 400
        duration_hours = (dt_end - dt_start).total_seconds() / 3600
        rate = 50 if connector_type == "CCS" else 30
        amount = round(duration_hours * rate, 2)
        user = User.query.filter_by(username=session["user"]).first()
        db.session.add(EVBooking(user_id=user.id, slot_id=slot.id, entry_time=start_time, exit_time=end_time, connector_type=connector_type, amount=amount, transaction_id=transaction_id or None, payment_status="verified", status="confirmed", booked_at=datetime.now().isoformat(timespec="minutes")))
        db.session.commit()
        return redirect("/my_bookings")
    entry_time = request.args.get("entry_time")
    exit_time = request.args.get("exit_time")
    next_booking = EVBooking.query.filter_by(slot_id=slot.id, status="confirmed").order_by(EVBooking.exit_time.desc()).first()
    next_available = next_booking.exit_time if next_booking else "Now"
    return render_template("book_ev.html", slot=slot, station={"name": station.location_name}, next_available=next_available, entry_time=entry_time, exit_time=exit_time)

@app.route("/ev/cancel/<int:slot_id>", methods=["POST"])
def ev_cancel(slot_id):
    if "user" not in session:
        return redirect("/login")
    user = User.query.filter_by(username=session["user"]).first()
    booking = EVBooking.query.filter_by(slot_id=slot_id, user_id=user.id, status="confirmed").order_by(EVBooking.id.desc()).first()
    if not booking:
        return "❌ Booking not found"
    refund_given = round(float(booking.amount or 0) / 3, 2)
    booking.status = "cancelled"
    booking.refund_amount = refund_given
    booking.cancelled_at = datetime.now().isoformat(timespec="minutes")
    booking.released_to_instant = True
    booking.instant_released_at = datetime.now().isoformat(timespec="minutes")
    user.wallet_balance += refund_given
    slot = EVSlot.query.get(slot_id)
    station = EVStation.query.get(slot.station_id)
    db.session.add(WalletTransaction(user_id=user.id, tx_type="EV Refund", reference_type="ev", reference_id=booking.id, location=station.location_name, amount=refund_given, created_at=datetime.now().isoformat(timespec="minutes")))
    db.session.commit()
    return redirect("/my_bookings")

@app.route("/carpool/rides")
def carpool_rides():
    if "user" not in session:
        return redirect("/login")
    rides = CarpoolRide.query.order_by(CarpoolRide.id.desc()).all()
    return render_template("carpool_rides.html", rides=rides)

@app.route("/carpool/book/<int:ride_id>", methods=["GET", "POST"])
def carpool_book(ride_id):
    if "user" not in session:
        return redirect("/login")
    ride = CarpoolRide.query.get(ride_id)
    if not ride:
        return {"error": "Ride not found"}, 404
    if request.method == "POST":
        seats_booked = int(request.form.get("seats_booked", 1) or 1)
        if ride.available_seats < seats_booked:
            return {"error": "No seats available"}, 400
        ride.available_seats -= seats_booked
        user = User.query.filter_by(username=session["user"]).first()
        db.session.add(CarpoolBooking(ride_id=ride.id, user_id=user.id, seats_booked=seats_booked, booked_at=datetime.now().isoformat(timespec="minutes")))
        db.session.commit()
        return redirect("/carpool/bookings")
    return render_template("ride_book.html", ride=ride)

@app.route("/carpool/bookings")
def carpool_bookings():
    if "user" not in session:
        return redirect("/login")
    user = User.query.filter_by(username=session["user"]).first()
    bookings = CarpoolBooking.query.filter_by(user_id=user.id).all()
    user_bookings = []
    for b in bookings:
        ride = CarpoolRide.query.get(b.ride_id)
        user_bookings.append({"ride_id": ride.id, "driver_name": ride.driver_name, "start_location": ride.start_location, "destination": ride.destination, "date_time": ride.date_time, "seats_booked": b.seats_booked, "price": ride.price})
    return render_template("carpool_bookings.html", carpool_bookings=user_bookings)

@app.route("/offer", methods=["GET", "POST"])
def offer():
    if "user" not in session:
        return redirect("/login")
    message = None
    rides = CarpoolRide.query.order_by(CarpoolRide.id.desc()).all()
    if request.method == "POST":
        start = request.form.get('start')
        destination = request.form.get('destination')
        vehicle_type = request.form.get('vehicle_type')
        s_lat, s_lng = get_lat_lng(start)
        d_lat, d_lng = get_lat_lng(destination)
        if s_lat is None or d_lat is None:
            message = "❌ Could not find location."
            return render_template("offer.html", rides=rides, message=message)
        total_distance = calculate_distance(s_lat, s_lng, d_lat, d_lng)
        ride_price = round(total_distance * PRICE_PER_KM.get(vehicle_type, 10))
        is_shared = request.form.get("is_shared") == "yes"
        db.session.add(CarpoolRide(driver_name=request.form.get('driver_name'), phone=request.form.get('phone'), vehicle_type=vehicle_type, vehicle=request.form.get('vehicle'), vehicle_number=request.form.get('vehicle_number'), start_location=start, destination=destination, date_time=request.form.get('time') or datetime.now().isoformat(timespec="minutes"), available_seats=int(request.form.get('seats') or 0), price=ride_price, notes=request.form.get('notes'), is_shared=is_shared, driver_lat=s_lat, driver_lng=s_lng, driver_dest_lat=d_lat, driver_dest_lng=d_lng))
        db.session.commit()
        message = f"✅ Ride added! Price: ₹{ride_price}"
        rides = CarpoolRide.query.order_by(CarpoolRide.id.desc()).all()
    return render_template("offer.html", rides=rides, message=message)

@app.route("/search", methods=["GET", "POST"])
def search():
    if "user" not in session:
        return redirect("/login")
    rides = []
    message = None
    if request.method == "POST":
        pickup = request.form.get("pickup")
        drop = request.form.get("drop")
        vehicle_type = request.form.get("vehicle_type")
        if not pickup or not drop or not vehicle_type:
            message = "Please fill all fields."
            return render_template("search.html", rides=[], message=message, PLACE_COORDINATES=PLACE_COORDINATES)
        p_lat, p_lng = get_lat_lng(pickup)
        d_lat, d_lng = get_lat_lng(drop)
        if p_lat is None or d_lat is None:
            message = "❌ Location not found."
            return render_template("search.html", rides=[], message=message, PLACE_COORDINATES=PLACE_COORDINATES)
        all_rides = CarpoolRide.query.filter_by(is_shared=True).all()
        filtered_rides = []
        for ride in all_rides:
            if ride.available_seats <= 0 or ride.vehicle_type != vehicle_type:
                continue
            if None in (ride.driver_lat, ride.driver_lng, ride.driver_dest_lat, ride.driver_dest_lng):
                continue
            exact_match = ride.start_location == pickup and ride.destination == drop
            dist_pickup = calculate_distance(p_lat, p_lng, ride.driver_lat, ride.driver_lng)
            dist_drop = calculate_distance(d_lat, d_lng, ride.driver_dest_lat, ride.driver_dest_lng)
            if exact_match or (dist_pickup <= 5 and dist_drop <= 5):
                total_dist = calculate_distance(ride.driver_lat, ride.driver_lng, ride.driver_dest_lat, ride.driver_dest_lng)
                user_dist = calculate_distance(p_lat, p_lng, d_lat, d_lng)
                ride.partial_price = max(1, round((user_dist / total_dist) * ride.price)) if total_dist > 0 else ride.price
                filtered_rides.append(ride)
        rides = filtered_rides
    return render_template("search.html", rides=rides, message=message, PLACE_COORDINATES=PLACE_COORDINATES)

@app.route("/my_bookings")
def my_bookings():
    if "user" not in session:
        return redirect("/login")
    user = User.query.filter_by(username=session["user"]).first()
    all_bookings = []
    for b in ParkingBooking.query.filter_by(user_id=user.id).all():
        slot = ParkingSlot.query.get(b.slot_id)
        area = ParkingArea.query.get(slot.area_id)
        all_bookings.append({"type": "Parking", "slot_id": slot.id, "station_name": area.location_name, "slot_number": slot.slot_number, "start_time": b.entry_time, "end_time": b.exit_time, "payment": f"₹{b.amount}", "amount": b.amount, "status": b.status, "refund_amount": b.refund_amount, "transaction_id": b.transaction_id or "", "payment_status": b.payment_status})
    for b in EVBooking.query.filter_by(user_id=user.id).all():
        slot = EVSlot.query.get(b.slot_id)
        station = EVStation.query.get(slot.station_id)
        all_bookings.append({"type": "EV", "slot_id": slot.id, "station_name": station.location_name, "slot_number": slot.slot_number, "start_time": b.entry_time, "end_time": b.exit_time, "payment": f"₹{b.amount}", "amount": b.amount, "status": b.status, "refund_amount": b.refund_amount, "transaction_id": b.transaction_id or "", "payment_status": b.payment_status})
    all_bookings.sort(key=lambda x: x.get("start_time", ""), reverse=True)
    return render_template("mybookings.html", bookings=all_bookings, wallet_balance=user.wallet_balance)

@app.route("/parking_bookings")
def parking_bookings():
    if "user" not in session:
        return redirect("/login")
    user = User.query.filter_by(username=session["user"]).first()
    bookings = ParkingBooking.query.filter_by(user_id=user.id).all()
    user_bookings = []
    for booking in bookings:
        slot = ParkingSlot.query.get(booking.slot_id)
        area = ParkingArea.query.get(slot.area_id)
        user_bookings.append({"location": area.location_name, "slot_id": slot.id, "entry_time": booking.entry_time, "exit_time": booking.exit_time, "payment": booking.payment})
    return render_template("parking_bookings.html", bookings=user_bookings, user=session["user"])

@app.route("/clear_all")
def clear_all():
    ParkingBooking.query.delete()
    EVBooking.query.delete()
    CarpoolBooking.query.delete()
    db.session.commit()
    return "✅ All bookings deleted"

if __name__ == "__main__":
    with app.app_context():
        # db.drop_all()
        db.create_all()
        seed_parking_areas()
        seed_ev_stations()
        seed_demo_carpool()
        calculate_model_accuracy()
    app.run(debug=True)
