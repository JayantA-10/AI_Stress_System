# app.py

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from ml_model import predict_stress
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stress_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ── DATABASE MODELS ───────────────────────────────────────────


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100))
    password = db.Column(db.String(200))
    role = db.Column(db.String(20), default="student")  # FIX: added role field
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class DailyStressLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # FIX: added FK
    study_hours = db.Column(db.Float)
    sleep_hours = db.Column(db.Float)
    mood_level = db.Column(db.Integer)
    assignment_pressure = db.Column(db.Integer)
    study_consistency = db.Column(db.Integer)
    performance_trend = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class StressPredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # FIX: added FK
    stress_prediction = db.Column(db.String(50))
    stress_confidence = db.Column(db.Float)
    burnout_risk = db.Column(db.Float)
    suggested_action = db.Column(db.String(300))
    alert_sent = db.Column(db.Boolean, default=False)  # FIX: added alert flag
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='predictions')


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── RULE-BASED AI LAYER ───────────────────────────────────────


def rule_based_logic(data, predicted_level, confidence):
    burnout_risk = 0

    # Sleep Deprivation
    if data['sleep_hours'] < 6:
        burnout_risk += 20

    # Overwork + Low Mood
    if data['study_hours'] > 6 and data['mood_level'] < 4:
        burnout_risk += 30

    # Declining Performance
    if data['performance_trend'] == -1:
        burnout_risk += 20

    # High Assignment Pressure
    if data['assignment_pressure'] > 8:
        burnout_risk += 25

    # Low Study Consistency
    if data['study_consistency'] < 4:
        burnout_risk += 15

    # Adjust Based on ML Confidence
    if predicted_level == "High" and confidence > 80:
        burnout_risk += 15

    # Cap at 100%
    burnout_risk = min(burnout_risk, 100)

    # Suggested Action Logic
    if burnout_risk > 70:
        suggestion = "Critical burnout risk. Immediate rest and academic counseling recommended."
    elif burnout_risk > 40:
        suggestion = "Moderate burnout risk. Improve sleep and reduce workload."
    else:
        suggestion = "Stable condition. Maintain healthy routine."

    return burnout_risk, suggestion


# ── ROUTES ────────────────────────────────────────────────────


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        role = request.form.get("role", "student")  # FIX: capture role from form

        # FIX: check for duplicate username
        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please choose another.")
            return redirect(url_for("register"))

        new_user = User(username=username, email=email, password=password, role=role)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration Successful! Please Login.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()

        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            # FIX: redirect counselors to their own dashboard
            if user.role == "counselor":
                return redirect(url_for("counselor_dashboard"))
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid Credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/daily_form", methods=["GET", "POST"])
@login_required
def daily_form():
    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        sleep_hours = float(request.form["sleep_hours"])
        mood_level = int(request.form["mood_level"])
        assignment_pressure = int(request.form["assignment_pressure"])
        study_consistency = int(request.form["study_consistency"])
        performance_trend = int(request.form["performance_trend"])

        # Save daily log
        log = DailyStressLog(
            user_id=current_user.id,
            study_hours=study_hours,
            sleep_hours=sleep_hours,
            mood_level=mood_level,
            assignment_pressure=assignment_pressure,
            study_consistency=study_consistency,
            performance_trend=performance_trend
        )
        db.session.add(log)
        db.session.commit()

        # ML Prediction
        features = [
            study_hours,
            sleep_hours,
            mood_level,
            assignment_pressure,
            study_consistency,
            performance_trend
        ]

        predicted_level, confidence = predict_stress(features)

        # Rule-based logic
        burnout, suggestion = rule_based_logic({
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "mood_level": mood_level,
            "assignment_pressure": assignment_pressure,
            "study_consistency": study_consistency,
            "performance_trend": performance_trend
        }, predicted_level, confidence)

        # FIX: auto-flag alert for high risk students
        alert_sent = (burnout > 70 or predicted_level == "High")

        # Save prediction
        result = StressPredictionResult(
            user_id=current_user.id,
            stress_prediction=predicted_level,
            stress_confidence=confidence,
            burnout_risk=burnout,
            suggested_action=suggestion,
            alert_sent=alert_sent
        )
        db.session.add(result)
        db.session.commit()

        return redirect(url_for("dashboard"))

    return render_template("daily_form.html")


@app.route("/dashboard")
@login_required
def dashboard():
    latest_result = StressPredictionResult.query.filter_by(
        user_id=current_user.id
    ).order_by(StressPredictionResult.created_at.desc()).first()

    # FIX: also pass recent history for the dashboard table
    history = StressPredictionResult.query.filter_by(
        user_id=current_user.id
    ).order_by(StressPredictionResult.created_at.desc()).limit(7).all()

    return render_template("dashboard.html", result=latest_result, history=history)


@app.route("/analytics")
@login_required
def analytics():
    stress_results = StressPredictionResult.query.filter_by(
        user_id=current_user.id
    ).order_by(StressPredictionResult.created_at.asc()).all()

    # FIX: was only passing stress_confidence — now passing full chart data
    chart_data = {
        "dates":      [r.created_at.strftime("%b %d") for r in stress_results],
        "burnout":    [r.burnout_risk for r in stress_results],
        "confidence": [r.stress_confidence for r in stress_results],
        "levels":     [r.stress_prediction for r in stress_results],
    }

    counts = {"Low": 0, "Moderate": 0, "High": 0}
    for r in stress_results:
        if r.stress_prediction in counts:
            counts[r.stress_prediction] += 1

    return render_template(
        "analytics.html",
        chart_data=chart_data,
        counts=counts,
        results=stress_results
    )


# ── COUNSELOR DASHBOARD (NEW) ─────────────────────────────────


@app.route("/counselor")
@login_required
def counselor_dashboard():
    # Guard: only counselors can access
    if current_user.role != "counselor":
        flash("Access denied. Counselors only.")
        return redirect(url_for("dashboard"))

    students = User.query.filter_by(role="student").all()
    student_data = []

    for student in students:
        latest = StressPredictionResult.query.filter_by(
            user_id=student.id
        ).order_by(StressPredictionResult.created_at.desc()).first()

        student_data.append({
            "id":       student.id,
            "username": student.username,
            "email":    student.email,
            "latest":   latest
        })

    # Sort: High risk first, then Moderate, then Low, then no data
    def risk_order(s):
        if not s["latest"]:
            return 3
        return {"High": 0, "Moderate": 1, "Low": 2}.get(s["latest"].stress_prediction, 3)

    student_data.sort(key=risk_order)

    high_risk_count = sum(1 for s in student_data if s["latest"] and s["latest"].stress_prediction == "High")
    alert_count     = sum(1 for s in student_data if s["latest"] and s["latest"].alert_sent)

    return render_template(
        "counselor.html",
        student_data=student_data,
        high_risk_count=high_risk_count,
        alert_count=alert_count
    )


@app.route("/counselor/student/<int:user_id>")
@login_required
def student_detail(user_id):
    if current_user.role != "counselor":
        flash("Access denied.")
        return redirect(url_for("dashboard"))

    student = User.query.get_or_404(user_id)
    history = StressPredictionResult.query.filter_by(
        user_id=user_id
    ).order_by(StressPredictionResult.created_at.asc()).all()

    chart_data = {
        "dates":   [r.created_at.strftime("%b %d") for r in history],
        "burnout": [r.burnout_risk for r in history],
        "levels":  [r.stress_prediction for r in history],
    }

    return render_template(
        "student_detail.html",
        student=student,
        history=history,
        chart_data=chart_data
    )


# ── INIT DATABASE ─────────────────────────────────────────────


if __name__ == "__main__":
    # FIX: always run db.create_all() — safe even if tables already exist
    with app.app_context():
        db.create_all()
        print("Database ready.")

    app.run(debug=True)