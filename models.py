# models.py
# Standalone DB model definitions — imported by app.py via db instance
# This file documents all database tables used in the system.
#
# NOTE: The actual db = SQLAlchemy(app) lives in app.py.
# These are shown here for reference/documentation purposes.
# If you want to split models into a separate file in future,
# initialise db in a separate extensions.py and import it here.

"""
TABLE: user
-----------
id            INTEGER  PRIMARY KEY
username      TEXT     UNIQUE — login name
email         TEXT     — contact email
password      TEXT     — hashed with werkzeug
role          TEXT     — 'student' or 'counselor'
created_at    DATETIME — auto-set on registration

TABLE: daily_stress_log
-----------------------
id                  INTEGER  PRIMARY KEY
user_id             INTEGER  FK → user.id
study_hours         FLOAT    — hours studied today
sleep_hours         FLOAT    — hours slept last night
mood_level          INTEGER  — 1 (very low) to 10 (great)
assignment_pressure INTEGER  — 1 (none) to 10 (overwhelming)
study_consistency   INTEGER  — 1 (irregular) to 10 (consistent)
performance_trend   INTEGER  — -1 (declining), 0 (stable), 1 (improving)
created_at          DATETIME — auto-set on log submission

TABLE: stress_prediction_result
--------------------------------
id                  INTEGER  PRIMARY KEY
user_id             INTEGER  FK → user.id
stress_prediction   TEXT     — 'Low', 'Moderate', or 'High'
stress_confidence   FLOAT    — ML model confidence (0-100%)
burnout_risk        FLOAT    — rule-based burnout score (0-100%)
suggested_action    TEXT     — AI-generated recommendation
alert_sent          BOOLEAN  — True if burnout > 70% or prediction = High
created_at          DATETIME — auto-set on prediction
"""