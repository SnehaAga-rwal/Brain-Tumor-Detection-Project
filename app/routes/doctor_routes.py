# ============================================
# FILE: app/routes/doctor_routes.py
# ============================================
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from app.database.models import Doctor, Patient, Scan, Diagnosis, Treatment, Report, Notification, AuditLog
from app.utils.email_notifications import send_treatment_advice
from datetime import datetime
import json

bp = Blueprint('doctor', __name__, url_prefix='/doctor')


@bp.route('/dashboard')
@login_required
def dashboard():
    """Doctor dashboard showing pending cases and statistics"""
    if not current_user.is_doctor():
        flash('Access denied. Doctor access only.', 'danger')
        return redirect(url_for('auth.login'))

    doctor = Doctor.query.filter_by(user_id=current_user.id).first()

    # Get pending reviews (scans with AI diagnosis but not verified by doctor)
    pending_diagnoses = Diagnosis.query.filter_by(doctor_verified=False) \
        .join(Scan) \
        .order_by(Diagnosis.created_at.asc()) \
        .all()

    # If doctor profile is missing, render dashboard with zeros
    if not doctor:
        critical_count = 0
        urgent_count = 0
        normal_count = 0
        total_reviewed = 0
        reviewed_today = 0
    else:
        # Get counts by severity
        critical_count = Treatment.query.filter_by(
            doctor_id=doctor.id,
            severity_level='critical'
        ).count()

        urgent_count = Treatment.query.filter_by(
            doctor_id=doctor.id,
            severity_level='urgent_surgery_needed'
        ).count()

        normal_count = Treatment.query.filter_by(
            doctor_id=doctor.id,
            severity_level='normal_monitoring'
        ).count()

        # Statistics
        total_reviewed = Diagnosis.query.filter_by(doctor_id=doctor.id).count()
        reviewed_today = Diagnosis.query.filter(
            Diagnosis.doctor_id == doctor.id,
            Diagnosis.verification_date >= datetime.utcnow().date()
        ).count()

    return render_template('doctor/dashboard.html',
                           doctor=doctor,
                           pending_diagnoses=pending_diagnoses,
                           critical_count=critical_count,
                           urgent_count=urgent_count,
                           normal_count=normal_count,
                           total_reviewed=total_reviewed,
                           reviewed_today=reviewed_today)


@bp.route('/review/<int:scan_id>')
@login_required
def review_scan(scan_id):
    """Review a specific scan"""
    if not current_user.is_doctor():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    scan = Scan.query.get_or_404(scan_id)
    diagnosis = Diagnosis.query.filter_by(scan_id=scan.id).first_or_404()

    # Get patient info
    patient = Patient.query.get(scan.patient_id)

    # Parse model predictions JSON (supports legacy flat dict or wrapped payload from ensemble)
    raw_mp = json.loads(diagnosis.model_predictions) if diagnosis.model_predictions else {}
    if isinstance(raw_mp, dict) and "model_details" in raw_mp:
        model_predictions = raw_mp.get("model_details") or {}
        prediction_meta = {k: v for k, v in raw_mp.items() if k != "model_details"}
    else:
        model_predictions = raw_mp
        prediction_meta = {}

    return render_template('doctor/review_scan.html',
                           doctor=doctor,
                           scan=scan,
                           diagnosis=diagnosis,
                           patient=patient,
                           model_predictions=model_predictions,
                           prediction_meta=prediction_meta)


@bp.route('/submit-review/<int:diagnosis_id>', methods=['POST'])
@login_required
def submit_review(diagnosis_id):
    """Submit doctor's review and treatment plan"""
    if not current_user.is_doctor():
        return jsonify({'error': 'Access denied'}), 403

    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    diagnosis = Diagnosis.query.get_or_404(diagnosis_id)

    try:
        data = request.json

        # Update diagnosis with doctor's verification
        diagnosis.doctor_id = doctor.id
        diagnosis.doctor_verified = True
        diagnosis.doctor_notes = data.get('doctor_notes', '')
        diagnosis.doctor_diagnosis = data.get('doctor_diagnosis', diagnosis.ai_prediction)
        diagnosis.verification_date = datetime.utcnow()

        # Create treatment plan
        treatment = Treatment(
            diagnosis_id=diagnosis.id,
            doctor_id=doctor.id,
            severity_level=data.get('severity_level', 'normal_monitoring'),
            follow_up_date=datetime.fromisoformat(data['follow_up_date']) if data.get('follow_up_date') else None,
            referred_specialist=data.get('referred_specialist', ''),
            additional_notes=data.get('additional_notes', '')
        )

        # Set encrypted fields
        treatment.prescription = data.get('prescription', '')
        treatment.recommended_therapies = data.get('recommended_therapies', [])
        treatment.lifestyle_modifications = data.get('lifestyle_modifications', [])

        db.session.add(treatment)

        # Log review
        log = AuditLog(
            user_id=current_user.id,
            action='DIAGNOSIS_REVIEWED',
            entity_type='diagnosis',
            entity_id=diagnosis.id,
            ip_address=request.remote_addr,
            details=f'Reviewed diagnosis. Severity: {treatment.severity_level}'
        )
        db.session.add(log)

        db.session.commit()

        # Send email notification to patient
        patient = Patient.query.get(diagnosis.scan.patient_id)
        send_treatment_advice(patient, doctor, diagnosis, treatment)

        return jsonify({'success': True, 'message': 'Review submitted successfully'})

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Review submission error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/patients')
@login_required
def patients_list():
    """View all patients assigned to doctor"""
    if not current_user.is_doctor():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    doctor = Doctor.query.filter_by(user_id=current_user.id).first()

    # Get all patients with diagnoses
    patients_data = []

    # In production, this would be based on actual assignments
    # For now, get all patients who have diagnoses
    diagnoses = Diagnosis.query.filter_by(doctor_id=doctor.id).all()
    patient_ids = set(d.scan.patient_id for d in diagnoses)

    for patient_id in patient_ids:
        patient = Patient.query.get(patient_id)
        latest_diagnosis = Diagnosis.query.join(Scan) \
            .filter(Scan.patient_id == patient_id) \
            .order_by(Diagnosis.created_at.desc()) \
            .first()

        latest_treatment = Treatment.query.filter_by(
            diagnosis_id=latest_diagnosis.id
        ).first() if latest_diagnosis else None

        patients_data.append({
            'patient': patient,
            'latest_diagnosis': latest_diagnosis,
            'latest_treatment': latest_treatment,
            'scan_count': Scan.query.filter_by(patient_id=patient_id).count()
        })

    return render_template('doctor/patients.html',
                           patients_data=patients_data)


@bp.route('/patient/<int:patient_id>')
@login_required
def patient_history(patient_id):
    """View complete history of a patient"""
    if not current_user.is_doctor():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    patient = Patient.query.get_or_404(patient_id)

    # Get all scans for this patient
    scans = Scan.query.filter_by(patient_id=patient.id) \
        .order_by(Scan.upload_date.desc()) \
        .all()

    history = []
    for scan in scans:
        diagnosis = Diagnosis.query.filter_by(scan_id=scan.id).first()
        treatment = Treatment.query.filter_by(diagnosis_id=diagnosis.id).first() if diagnosis else None

        history.append({
            'scan': scan,
            'diagnosis': diagnosis,
            'treatment': treatment
        })

    return render_template('doctor/patient_history.html',
                           patient=patient,
                           history=history)


@bp.route('/reports')
@login_required
def reports():
    """View generated reports"""
    if not current_user.is_doctor():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    doctor = Doctor.query.filter_by(user_id=current_user.id).first()

    # Get all diagnoses by this doctor with their reports
    diagnoses = Diagnosis.query.filter_by(doctor_id=doctor.id) \
        .order_by(Diagnosis.verification_date.desc()) \
        .all()

    reports_data = []
    for diagnosis in diagnoses:
        report = Report.query.filter_by(diagnosis_id=diagnosis.id).first()
        if report:
            reports_data.append({
                'diagnosis': diagnosis,
                'report': report,
                'patient': diagnosis.scan.patient
            })

    return render_template('doctor/reports.html',
                           reports_data=reports_data)


@bp.route('/analytics')
@login_required
def analytics():
    """View personal analytics"""
    if not current_user.is_doctor():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    doctor = Doctor.query.filter_by(user_id=current_user.id).first()

    # Calculate statistics
    total_diagnoses = Diagnosis.query.filter_by(doctor_id=doctor.id).count()

    # Accuracy compared to AI
    diagnoses = Diagnosis.query.filter_by(doctor_id=doctor.id).all()
    agreed = sum(1 for d in diagnoses if d.doctor_diagnosis == d.ai_prediction)
    accuracy = (agreed / total_diagnoses * 100) if total_diagnoses > 0 else 0

    # Severity distribution
    treatments = Treatment.query.filter_by(doctor_id=doctor.id).all()
    severity_counts = {
        'critical': 0,
        'urgent_surgery_needed': 0,
        'normal_monitoring': 0
    }

    for t in treatments:
        if t.severity_level in severity_counts:
            severity_counts[t.severity_level] += 1

    # Monthly trends (last 6 months)
    from datetime import timedelta
    import numpy as np

    months = []
    monthly_counts = []

    for i in range(5, -1, -1):
        date = datetime.utcnow() - timedelta(days=30 * i)
        month_str = date.strftime('%b %Y')
        months.append(month_str)

        count = Diagnosis.query.filter(
            Diagnosis.doctor_id == doctor.id,
            Diagnosis.verification_date >= date - timedelta(days=30),
            Diagnosis.verification_date < date
        ).count()
        monthly_counts.append(count)

    return render_template('doctor/analytics.html',
                           doctor=doctor,
                           total_diagnoses=total_diagnoses,
                           accuracy=accuracy,
                           severity_counts=severity_counts,
                           months=months,
                           monthly_counts=monthly_counts)

