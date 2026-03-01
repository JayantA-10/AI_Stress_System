# app.py

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from ml_model import predict_stress
import os
import requests as http_requests

# ── Google OAuth ──────────────────────────────────────────────
from authlib.integrations.flask_client import OAuth

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'supersecretkey')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stress_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Google OAuth credentials (set these as environment variables)
GOOGLE_CLIENT_ID     = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ── DATABASE MODELS ───────────────────────────────────────────


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100))
    password = db.Column(db.String(200), nullable=True)   # nullable: Google users have no password
    role = db.Column(db.String(20), default="student")
    google_id = db.Column(db.String(200), unique=True, nullable=True)  # Google OAuth ID
    avatar = db.Column(db.String(500), nullable=True)     # Google profile picture URL
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


# ── AUTO PERFORMANCE TREND CALCULATOR ────────────────────────

def auto_performance_trend(user_id):
    """
    Automatically calculates performance trend by comparing
    this week's average mood & study consistency vs last week's.

    Returns:
         1  → Improving
         0  → Stable
        -1  → Declining
    """
    from datetime import timedelta

    now = datetime.utcnow()
    this_week_start = now - timedelta(days=7)
    last_week_start = now - timedelta(days=14)

    # This week's logs (last 7 days)
    this_week = DailyStressLog.query.filter(
        DailyStressLog.user_id == user_id,
        DailyStressLog.created_at >= this_week_start
    ).all()

    # Last week's logs (7–14 days ago)
    last_week = DailyStressLog.query.filter(
        DailyStressLog.user_id == user_id,
        DailyStressLog.created_at >= last_week_start,
        DailyStressLog.created_at < this_week_start
    ).all()

    # Not enough history yet — default to Stable
    if not this_week or not last_week:
        return 0, "Stable (not enough history yet)"

    # Score = average of mood_level + study_consistency (higher = better performance)
    def avg_score(logs):
        return sum(l.mood_level + l.study_consistency for l in logs) / len(logs)

    this_score = avg_score(this_week)
    last_score = avg_score(last_week)
    diff = this_score - last_score

    if diff > 1.5:
        return 1, f"Improving (this week: {round(this_score,1)}, last week: {round(last_score,1)})"
    elif diff < -1.5:
        return -1, f"Declining (this week: {round(this_score,1)}, last week: {round(last_score,1)})"
    else:
        return 0, f"Stable (this week: {round(this_score,1)}, last week: {round(last_score,1)})"


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

        if user and user.password and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            if user.role == "counselor":
                return redirect(url_for("counselor_dashboard"))
            return redirect(url_for("dashboard"))
        elif user and not user.password:
            flash("This account uses Google Sign-In. Please click 'Sign in with Google'.")
        else:
            flash("Invalid Credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))



# ── GOOGLE OAUTH ROUTES ───────────────────────────────────────

@app.route("/login/google")
def google_login():
    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/login/google/callback")
def google_callback():
    try:
        token = google.authorize_access_token()
    except Exception as e:
        flash(f"Google login failed: {str(e)}")
        return redirect(url_for("login"))

    user_info = token.get("userinfo")
    if not user_info:
        flash("Could not get user info from Google.")
        return redirect(url_for("login"))

    google_id = user_info.get("sub")
    email     = user_info.get("email")
    name      = user_info.get("name", email.split("@")[0])
    avatar    = user_info.get("picture", "")

    # Check if user already exists by google_id or email
    user = User.query.filter_by(google_id=google_id).first()
    if not user:
        user = User.query.filter_by(email=email).first()

    if user:
        # Existing user — update Google fields if missing
        if not user.google_id:
            user.google_id = google_id
        if not user.avatar:
            user.avatar = avatar
        db.session.commit()
    else:
        # Brand new user — create account automatically
        username = name.replace(" ", "").lower()
        # Ensure username is unique
        base = username
        counter = 1
        while User.query.filter_by(username=username).first():
            username = f"{base}{counter}"
            counter += 1

        user = User(
            username=username,
            email=email,
            password=None,
            role="student",
            google_id=google_id,
            avatar=avatar,
        )
        db.session.add(user)
        db.session.commit()

    login_user(user)
    if user.role == "counselor":
        return redirect(url_for("counselor_dashboard"))
    return redirect(url_for("dashboard"))

@app.route("/daily_form", methods=["GET", "POST"])
@login_required
def daily_form():
    # AUTO: calculate performance trend to show on the form before submit
    performance_trend, trend_label = auto_performance_trend(current_user.id)

    if request.method == "POST":
        study_hours        = float(request.form["study_hours"])
        sleep_hours        = float(request.form["sleep_hours"])
        mood_level         = int(request.form["mood_level"])
        assignment_pressure= int(request.form["assignment_pressure"])
        study_consistency  = int(request.form["study_consistency"])

        # AUTO: recalculate trend at submit time (not from form input)
        performance_trend, trend_label = auto_performance_trend(current_user.id)

        # Save daily log with auto-calculated trend
        log = DailyStressLog(
            user_id=current_user.id,
            study_hours=study_hours,
            sleep_hours=sleep_hours,
            mood_level=mood_level,
            assignment_pressure=assignment_pressure,
            study_consistency=study_consistency,
            performance_trend=performance_trend   # auto-calculated
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
            "study_hours":         study_hours,
            "sleep_hours":         sleep_hours,
            "mood_level":          mood_level,
            "assignment_pressure": assignment_pressure,
            "study_consistency":   study_consistency,
            "performance_trend":   performance_trend
        }, predicted_level, confidence)

        # Auto-flag alert for high risk students
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

    # Pass trend info to the form so student can see what was detected
    return render_template(
        "daily_form.html",
        performance_trend=performance_trend,
        trend_label=trend_label
    )


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

    # Sort by stress level group (High→Moderate→Low→No data)
    # then by burnout % descending within each group
    # so the most at-risk student is always at the very top
    def risk_order(s):
        if not s["latest"]:
            return (3, 0)
        level_rank   = {"High": 0, "Moderate": 1, "Low": 2}.get(s["latest"].stress_prediction, 3)
        burnout_rank = -s["latest"].burnout_risk   # negative so higher burnout sorts first
        return (level_rank, burnout_rank)

    student_data.sort(key=risk_order)

    # Attach priority rank number for display in the table
    for i, s in enumerate(student_data):
        s["rank"] = i + 1

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




# ── AI COUNSELOR CHAT ─────────────────────────────────────────


@app.route("/chat")
@login_required
def chat():
    # Fetch student's latest stress data to give AI context
    latest = StressPredictionResult.query.filter_by(
        user_id=current_user.id
    ).order_by(StressPredictionResult.created_at.desc()).first()

    return render_template("chat.html", latest=latest)


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():

    data = request.get_json()
    messages = data.get("messages", [])

    # Fetch student's latest stress result for context
    latest = StressPredictionResult.query.filter_by(
        user_id=current_user.id
    ).order_by(StressPredictionResult.created_at.desc()).first()

    # Fetch last 5 daily logs for extra context
    recent_logs = DailyStressLog.query.filter_by(
        user_id=current_user.id
    ).order_by(DailyStressLog.created_at.desc()).limit(5).all()

    # Build student context summary for the AI
    if latest:
        stress_context = f"""
Current Student Profile:
- Name: {current_user.username}
- Latest Stress Level: {latest.stress_prediction}
- Burnout Risk: {latest.burnout_risk}%
- Model Confidence: {latest.stress_confidence}%
- Last Recommendation: {latest.suggested_action}
- Alert Triggered: {"Yes" if latest.alert_sent else "No"}
"""
    else:
        stress_context = f"Student {current_user.username} has not logged any stress data yet."

    if recent_logs:
        log_lines = []
        for log in recent_logs:
            log_lines.append(
                f"  - {log.created_at.strftime('%b %d')}: "
                f"Study={log.study_hours}h, Sleep={log.sleep_hours}h, "
                f"Mood={log.mood_level}/10, Pressure={log.assignment_pressure}/10"
            )
        stress_context += "\nRecent Daily Logs:\n" + "\n".join(log_lines)

    system_prompt = f"""You are a warm, empathetic AI student counselor for an academic stress monitoring system called StressAI.

Your role is to:
- Provide emotional support and practical advice to students experiencing academic stress
- Suggest evidence-based coping strategies (breathing exercises, time management, sleep hygiene)
- Recommend professional help when stress levels are critically high
- Be encouraging, non-judgmental, and student-friendly
- Keep responses concise and actionable (2-4 sentences unless more detail is needed)
- Never diagnose medical conditions — always recommend seeing a professional for serious concerns

You have access to this student's real stress data:
{stress_context}

Use this data naturally in your responses — acknowledge their stress level, reference their recent patterns, and tailor advice to their specific situation. If their burnout risk is above 70%, gently but clearly encourage them to seek real counseling support."""

    # ── Get API key from environment variable ──
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"reply": "⚠️ API key not set. Please add your ANTHROPIC_API_KEY to your environment variables."}), 500

    try:
        response = http_requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": messages
            },
            timeout=30
        )

        # Check for API errors
        if response.status_code != 200:
            error_info = response.json()
            error_msg = error_info.get("error", {}).get("message", "Unknown API error")
            return jsonify({"reply": f"API Error: {error_msg}"}), 500

        result = response.json()
        reply = result["content"][0]["text"]
        return jsonify({"reply": reply})

    except http_requests.exceptions.Timeout:
        return jsonify({"reply": "The request timed out. Please try again."}), 500
    except http_requests.exceptions.ConnectionError:
        return jsonify({"reply": "Cannot connect to the AI service. Please check your internet connection."}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full error in terminal
        return jsonify({"reply": f"An error occurred: {str(e)}"}), 500

# ── INIT DATABASE ─────────────────────────────────────────────


if __name__ == "__main__":
    # FIX: always run db.create_all() — safe even if tables already exist
    with app.app_context():
        db.create_all()
        print("Database ready.")

    app.run(debug=True)