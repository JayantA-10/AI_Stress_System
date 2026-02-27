# app.py

from flask import Flask, render_template, redirect, url_for, request, flash
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


# DATABASE MODELS


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100))
    password = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class DailyStressLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    study_hours = db.Column(db.Float)
    sleep_hours = db.Column(db.Float)
    mood_level = db.Column(db.Integer)
    assignment_pressure = db.Column(db.Integer)
    study_consistency = db.Column(db.Integer)
    performance_trend = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class StressPredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    stress_prediction = db.Column(db.String(50))
    stress_confidence = db.Column(db.Float)
    burnout_risk = db.Column(db.Float)
    suggested_action = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



# RULE-BASED AI LAYER


def rule_based_logic(data, predicted_level, confidence):
    burnout_risk = 0
    stress_weight_adjustment = 0

    
    # Sleep Deprivation
    
    if data['sleep_hours'] < 6:
        burnout_risk += 20
        stress_weight_adjustment += 5

    
    #  Overwork + Low Mood
    
    if data['study_hours'] > 6 and data['mood_level'] < 4:
        burnout_risk += 30
        stress_weight_adjustment += 10

    
    #  Declining Performance
    
    if data['performance_trend'] == -1:
        burnout_risk += 20

    
    #  High Assignment Pressure
    
    if data['assignment_pressure'] > 8:
        burnout_risk += 25

    
    #  Low Study Consistency
    
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


# ROUTES


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        new_user = User(username=username, email=email, password=password)
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

        # Save prediction
        result = StressPredictionResult(
            user_id=current_user.id,
            stress_prediction=predicted_level,
            stress_confidence=confidence,
            burnout_risk=burnout,
            suggested_action=suggestion
        )
        db.session.add(result)
        db.session.commit()

        return redirect(url_for("dashboard"))

    return render_template("daily_form.html")


@app.route("/dashboard")
@login_required
def dashboard():
    latest_result = StressPredictionResult.query.filter_by(user_id=current_user.id).order_by(
        StressPredictionResult.created_at.desc()
    ).first()

    return render_template("dashboard.html", result=latest_result)


@app.route("/analytics")
@login_required
def analytics():
    stress_results = StressPredictionResult.query.filter_by(
        user_id=current_user.id
    ).all()

    stress_levels = [r.stress_confidence for r in stress_results]

    return render_template("analytics.html", stress_levels=stress_levels)



# INIT DATABASE


if __name__ == "__main__":
    if not os.path.exists("stress_system.db"):
        with app.app_context():
            db.create_all()
            print("Database Created")

    app.run(debug=True)