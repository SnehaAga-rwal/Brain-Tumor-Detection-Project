# ============================================
# FILE: app/routes/patient_routes.py (UPDATED)
# ============================================
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, send_file, jsonify
from flask_login import login_required, current_user
from app import db
from app.database.models import Patient, Scan, Diagnosis, Treatment, Report, Notification, AuditLog, Doctor
from app.utils.encryption import generate_secure_filename, hash_file
from app.utils.email_notifications import send_upload_confirmation, send_ai_analysis_complete, send_urgent_case_alert
from app.models.ensemble_model import get_ensemble_predictor
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json
import threading
import traceback

bp = Blueprint('patient', __name__, url_prefix='/patient')

# Shared AI predictor (loads weights once per process)
def _predictor():
    return get_ensemble_predictor()


@bp.route('/mark-all-read', methods=['POST'])
@login_required
def mark_all_read():
    """Mark all notifications as read for the current user"""
    try:
        notifications = Notification.query.filter_by(user_id=current_user.id, read_status=False).all()
        for n in notifications:
            n.read_status = True
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        current_app.logger.error(f"Error marking all read: {e}")
        return jsonify({'success': False}), 500


@bp.route('/dashboard')
@login_required
def dashboard():
    """Patient dashboard showing overview and recent scans"""
    if not current_user.is_patient():
        flash('Access denied. Patient access only.', 'danger')
        return redirect(url_for('auth.login'))

    patient = Patient.query.filter_by(user_id=current_user.id).first()

    if not patient:
        flash('Patient profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    # Get recent scans
    recent_scans = Scan.query.filter_by(patient_id=patient.id) \
        .order_by(Scan.upload_date.desc()) \
        .limit(5).all()

    # Get unread notifications
    unread_count = Notification.query.filter_by(
        user_id=current_user.id,
        read_status=False
    ).count()

    # Statistics
    total_scans = Scan.query.filter_by(patient_id=patient.id).count()
    completed_scans = Diagnosis.query.join(Scan).filter(
        Scan.patient_id == patient.id,
        Diagnosis.doctor_verified == True
    ).count()
    pending_scans = total_scans - completed_scans

    return render_template('patient/dashboard.html',
                           patient=patient,
                           recent_scans=recent_scans,
                           unread_count=unread_count,
                           total_scans=total_scans,
                           completed_scans=completed_scans,
                           pending_scans=pending_scans)


@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_scan():
    """Upload MRI scan for analysis"""
    if not current_user.is_patient():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    patient = Patient.query.filter_by(user_id=current_user.id).first()

    if not patient:
        flash('Patient profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    if request.method == 'POST':
        # Check if file was uploaded
        if 'mri_file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['mri_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}
        if '.' not in file.filename or \
                file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or DICOM files.'}), 400

        try:
            # Secure filename and save
            original_filename = secure_filename(file.filename)
            secure_name = generate_secure_filename(original_filename)

            upload_folder = current_app.config['UPLOAD_FOLDER']
            patient_folder = os.path.join(upload_folder, f"patient_{patient.id}")
            os.makedirs(patient_folder, exist_ok=True)

            file_path = os.path.join(patient_folder, secure_name)
            file.save(file_path)

            # Calculate file hash for integrity
            with open(file_path, 'rb') as f:
                file_hash = hash_file(f.read())

            # Create scan record
            scan = Scan(
                patient_id=patient.id,
                filename=original_filename,
                file_path=file_path,
                status='pending',
                file_size=os.path.getsize(file_path),
                file_hash=file_hash,
                image_metadata=json.dumps({
                    'original_filename': original_filename,
                    'secure_filename': secure_name,
                    'upload_time': datetime.utcnow().isoformat()
                })
            )

            db.session.add(scan)
            db.session.flush()

            # Log upload
            log = AuditLog(
                user_id=current_user.id,
                action='SCAN_UPLOADED',
                entity_type='scan',
                entity_id=scan.id,
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                details=f'Uploaded scan: {original_filename}'
            )
            db.session.add(log)

            db.session.commit()

            # Send confirmation email (in background)
            try:
                send_upload_confirmation(patient, scan)
            except Exception as e:
                current_app.logger.error(f"Email error: {e}")

            # Start AI analysis in background with app context
            def process_scan_with_context(app, scan_id):
                with app.app_context():
                    try:
                        process_scan_ai(scan_id)
                    except Exception as e:
                        current_app.logger.error(f"AI processing error: {e}\n{traceback.format_exc()}")

            thread = threading.Thread(
                target=process_scan_with_context,
                args=(current_app._get_current_object(), scan.id)
            )
            thread.daemon = True
            thread.start()

            return jsonify({
                'success': True,
                'scan_id': scan.id,
                'message': 'Scan uploaded successfully'
            })

        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500

    return render_template('patient/upload.html')


def process_scan_ai(scan_id):
    """Background task to process scan with AI"""
    try:
        scan = Scan.query.get(scan_id)
        if not scan:
            current_app.logger.error(f"Scan {scan_id} not found")
            return

        # Update status
        scan.status = 'processing'
        db.session.commit()

        # Run prediction (includes per-class probabilities and per-model softmax outputs)
        result = _predictor().predict(scan.file_path)

        current_app.logger.info(
            "Scan %s AI: pred=%s conf=%.2f probs=%s models=%s err=%s",
            scan_id,
            result.get("prediction"),
            float(result.get("confidence") or 0),
            result.get("class_probabilities"),
            result.get("models_used"),
            result.get("error"),
        )

        diagnosis_payload = {
            "model_details": result.get("model_details", {}),
            "class_probabilities": result.get("class_probabilities", {}),
            "all_probabilities": result.get("all_probabilities", {}),
            "models_used": result.get("models_used", []),
            "preprocessing": result.get("preprocessing"),
            "error": result.get("error"),
            "debug": result.get("debug"),
        }

        # Create diagnosis record
        diagnosis = Diagnosis(
            scan_id=scan.id,
            ai_prediction=result['prediction'],
            ai_confidence=result['confidence'],
            model_predictions=json.dumps(diagnosis_payload),
            ensemble_confidence=result.get('ensemble_confidence', result['confidence'])
        )

        db.session.add(diagnosis)
        scan.status = 'completed'

        # Log AI completion
        log = AuditLog(
            user_id=scan.patient.user_id,
            action='AI_ANALYSIS_COMPLETE',
            entity_type='diagnosis',
            entity_id=diagnosis.id,
            details=f'AI prediction: {result["prediction"]}'
        )
        db.session.add(log)

        db.session.commit()

        # Send notification to patient
        try:
            send_ai_analysis_complete(scan.patient, scan, diagnosis)
        except Exception as e:
            current_app.logger.error(f"Email error: {e}")

        # Check if urgent and notify doctor (tumor classes only; notumor/normal are non-urgent)
        pred = result.get('prediction')
        if pred not in ('notumor', 'normal', None) and result.get('confidence', 0) > 85:
            # Find doctor (in production, use patient's assigned doctor)
            doctor = Doctor.query.first()
            if doctor:
                try:
                    send_urgent_case_alert(doctor, scan.patient, diagnosis)
                except Exception as e:
                    current_app.logger.error(f"Urgent email error: {e}")

    except Exception as e:
        current_app.logger.error(f"AI processing error for scan {scan_id}: {e}\n{traceback.format_exc()}")
        if scan:
            scan.status = 'failed'
            db.session.commit()


@bp.route('/scan/<int:scan_id>')
@login_required
def scan_status(scan_id):
    """View scan status and results"""
    if not current_user.is_patient():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    patient = Patient.query.filter_by(user_id=current_user.id).first()

    if not patient:
        flash('Patient profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    scan = Scan.query.filter_by(id=scan_id, patient_id=patient.id).first_or_404()

    diagnosis = Diagnosis.query.filter_by(scan_id=scan.id).first()
    treatment = Treatment.query.filter_by(diagnosis_id=diagnosis.id).first() if diagnosis else None

    return render_template('patient/scan_result.html',
                           scan=scan,
                           diagnosis=diagnosis,
                           treatment=treatment)


@bp.route('/history')
@login_required
def history():
    """View complete medical history"""
    if not current_user.is_patient():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    patient = Patient.query.filter_by(user_id=current_user.id).first()

    if not patient:
        flash('Patient profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    # Get all scans with diagnoses
    scans = Scan.query.filter_by(patient_id=patient.id) \
        .order_by(Scan.upload_date.desc()) \
        .all()

    history_data = []
    for scan in scans:
        diagnosis = Diagnosis.query.filter_by(scan_id=scan.id).first()
        treatment = Treatment.query.filter_by(diagnosis_id=diagnosis.id).first() if diagnosis else None

        history_data.append({
            'scan': scan,
            'diagnosis': diagnosis,
            'treatment': treatment
        })

    return render_template('patient/history.html',
                           history_data=history_data)


@bp.route('/treatment/<int:diagnosis_id>')
@login_required
def view_treatment(diagnosis_id):
    """View treatment plan"""
    if not current_user.is_patient():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    patient = Patient.query.filter_by(user_id=current_user.id).first()

    if not patient:
        flash('Patient profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    diagnosis = Diagnosis.query.get_or_404(diagnosis_id)

    # Verify patient owns this diagnosis
    if diagnosis.scan.patient_id != patient.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('patient.dashboard'))

    treatment = Treatment.query.filter_by(diagnosis_id=diagnosis.id).first_or_404()

    return render_template('patient/treatment.html',
                           diagnosis=diagnosis,
                           treatment=treatment)


@bp.route('/download-report/<int:diagnosis_id>')
@login_required
def download_report(diagnosis_id):
    """Download PDF report"""
    if not current_user.is_patient():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    patient = Patient.query.filter_by(user_id=current_user.id).first()

    if not patient:
        flash('Patient profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    diagnosis = Diagnosis.query.get_or_404(diagnosis_id)

    # Verify patient owns this diagnosis
    if diagnosis.scan.patient_id != patient.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('patient.dashboard'))

    report = Report.query.filter_by(diagnosis_id=diagnosis.id).first()

    if not report:
        # Generate report if not exists
        from app.reporting.pdf_generator import PDFGenerator
        pdf_gen = PDFGenerator()
        report_path = pdf_gen.generate_patient_report(diagnosis.id)
        report = Report(
            diagnosis_id=diagnosis.id,
            pdf_path=report_path
        )
        db.session.add(report)
        db.session.commit()

    # Update access count
    report.access_count += 1
    report.last_accessed = datetime.utcnow()
    db.session.commit()

    return send_file(report.pdf_path, as_attachment=True)


@bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """View and edit patient profile"""
    if not current_user.is_patient():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    patient = Patient.query.filter_by(user_id=current_user.id).first()

    if not patient:
        flash('Patient profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    if request.method == 'POST':
        try:
            # Update profile (encrypted fields handled automatically)
            patient.phone = request.form.get('phone')
            patient.date_of_birth = request.form.get('date_of_birth')
            patient.address = request.form.get('address')
            patient.blood_group = request.form.get('blood_group')
            patient.emergency_contact = request.form.get('emergency_contact')

            # Handle allergies and conditions as JSON
            allergies = request.form.getlist('allergies')
            if allergies:
                patient.allergies = json.dumps(allergies)
            else:
                patient.allergies = None

            conditions = request.form.getlist('chronic_conditions')
            if conditions:
                patient.chronic_conditions = json.dumps(conditions)
            else:
                patient.chronic_conditions = None

            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('patient.profile'))
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Profile update error: {e}")
            flash('Error updating profile.', 'danger')

    # Decode JSON fields for display
    allergies_list = []
    if patient.allergies:
        try:
            allergies_list = json.loads(patient.allergies)
        except:
            allergies_list = []

    conditions_list = []
    if patient.chronic_conditions:
        try:
            conditions_list = json.loads(patient.chronic_conditions)
        except:
            conditions_list = []

    return render_template('patient/profile.html',
                           patient=patient,
                           allergies_list=allergies_list,
                           conditions_list=conditions_list)


@bp.route('/notifications')
@login_required
def notifications():
    """View all notifications"""
    if not current_user.is_patient():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    notifications_list = Notification.query.filter_by(user_id=current_user.id) \
        .order_by(Notification.created_at.desc()) \
        .all()

    return render_template('patient/notifications.html',
                           notifications=notifications_list)


@bp.route('/mark-notification-read/<int:notification_id>')
@login_required
def mark_notification_read(notification_id):
    """Mark notification as read"""
    notification = Notification.query.filter_by(
        id=notification_id,
        user_id=current_user.id
    ).first_or_404()

    notification.read_status = True
    db.session.commit()

    return redirect(request.referrer or url_for('patient.notifications'))
