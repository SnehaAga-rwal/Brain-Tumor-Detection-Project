# ============================================
# FILE: app/routes/admin_routes.py
# ============================================
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from app.database.models import User, Patient, Doctor, Scan, Diagnosis, Treatment, AuditLog, Notification
from datetime import datetime, timedelta
import json
import psutil
import platform

bp = Blueprint('admin', __name__, url_prefix='/admin')


@bp.route('/dashboard')
@login_required
def dashboard():
    """Admin dashboard with system overview"""
    if not current_user.is_admin():
        flash('Access denied. Admin access only.', 'danger')
        return redirect(url_for('auth.login'))

    # User statistics
    total_users = User.query.count()
    total_patients = User.query.filter_by(role='patient').count()
    total_doctors = User.query.filter_by(role='doctor').count()
    pending_doctors = Doctor.query.filter_by(verification_status=False).count()

    # Scan statistics
    total_scans = Scan.query.count()
    scans_today = Scan.query.filter(
        Scan.upload_date >= datetime.utcnow().date()
    ).count()

    # System health (simulated for demo)
    system_health = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'platform': platform.platform(),
        'python_version': platform.python_version()
    }

    # Recent activity
    recent_logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(10).all()

    return render_template('admin/dashboard.html',
                           total_users=total_users,
                           total_patients=total_patients,
                           total_doctors=total_doctors,
                           pending_doctors=pending_doctors,
                           total_scans=total_scans,
                           scans_today=scans_today,
                           system_health=system_health,
                           recent_logs=recent_logs)


@bp.route('/users')
@login_required
def users():
    """User management"""
    if not current_user.is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=users)


@bp.route('/user/<int:user_id>')
@login_required
def user_detail(user_id):
    """View user details"""
    if not current_user.is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    user = User.query.get_or_404(user_id)

    if user.is_patient():
        profile = Patient.query.filter_by(user_id=user.id).first()
        scans = Scan.query.filter_by(patient_id=profile.id).all() if profile else []
        return render_template('admin/user_patient.html', user=user, profile=profile, scans=scans)

    elif user.is_doctor():
        profile = Doctor.query.filter_by(user_id=user.id).first()
        diagnoses = Diagnosis.query.filter_by(doctor_id=profile.id).all() if profile else []
        return render_template('admin/user_doctor.html', user=user, profile=profile, diagnoses=diagnoses)

    else:
        return render_template('admin/user_admin.html', user=user)


@bp.route('/user/<int:user_id>/toggle-status', methods=['POST'])
@login_required
def toggle_user_status(user_id):
    """Enable/disable user account"""
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403

    user = User.query.get_or_404(user_id)

    # Toggle lock status
    if user.account_locked_until and user.account_locked_until > datetime.utcnow():
        user.account_locked_until = None  # Unlock
        message = f'User {user.username} unlocked'
    else:
        user.account_locked_until = datetime.utcnow() + timedelta(days=365)  # Lock for a year
        message = f'User {user.username} locked'

    # Log action
    log = AuditLog(
        user_id=current_user.id,
        action='USER_STATUS_TOGGLED',
        entity_type='user',
        entity_id=user.id,
        ip_address=request.remote_addr,
        details=message
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({'success': True, 'message': message})


@bp.route('/doctors/pending')
@login_required
def pending_doctors():
    """View pending doctor verifications"""
    if not current_user.is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    pending = Doctor.query.filter_by(verification_status=False).all()
    return render_template('admin/pending_doctors.html', pending_doctors=pending)


@bp.route('/doctor/<int:doctor_id>/verify', methods=['POST'])
@login_required
def verify_doctor(doctor_id):
    """Verify doctor credentials"""
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403

    doctor = Doctor.query.get_or_404(doctor_id)
    action = request.json.get('action')

    if action == 'approve':
        doctor.verification_status = True
        doctor.verification_date = datetime.utcnow()
        message = f'Doctor {doctor.user.username} approved'

        # Notify doctor via email (simplified)
        notification = Notification(
            user_id=doctor.user.id,
            type='email',
            subject='Your Doctor Account Has Been Verified',
            message='Your doctor account has been approved. You can now log in and start reviewing cases.',
            sent_status=True,
            sent_at=datetime.utcnow()
        )
        db.session.add(notification)

    else:  # reject
        # Optionally delete or flag for more info
        message = f'Doctor {doctor.user.username} rejected'

    # Log action
    log = AuditLog(
        user_id=current_user.id,
        action='DOCTOR_VERIFIED',
        entity_type='doctor',
        entity_id=doctor.id,
        ip_address=request.remote_addr,
        details=message
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({'success': True, 'message': message})


@bp.route('/audit-logs')
@login_required
def audit_logs():
    """View audit logs"""
    if not current_user.is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    page = request.args.get('page', 1, type=int)
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).paginate(page=page, per_page=50)

    return render_template('admin/audit_logs.html', logs=logs)


@bp.route('/system/monitor')
@login_required
def system_monitor():
    """Real-time system monitoring"""
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403

    # Collect system metrics
    metrics = {
        'cpu': {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent,
            'used': psutil.virtual_memory().used
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'used': psutil.disk_usage('/').used,
            'free': psutil.disk_usage('/').free,
            'percent': psutil.disk_usage('/').percent
        },
        'database': {
            'users': User.query.count(),
            'scans': Scan.query.count(),
            'diagnoses': Diagnosis.query.count()
        },
        'recent_actions': AuditLog.query.filter(
            AuditLog.timestamp >= datetime.utcnow() - timedelta(minutes=5)
        ).count()
    }

    return jsonify(metrics)


@bp.route('/backup', methods=['POST'])
@login_required
def create_backup():
    """Trigger database backup"""
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403

    # In production, this would trigger actual backup
    # For now, just log the action

    log = AuditLog(
        user_id=current_user.id,
        action='BACKUP_CREATED',
        ip_address=request.remote_addr,
        details='Manual database backup triggered'
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Backup initiated'})


@bp.route('/iot/devices')
@login_required
def iot_devices():
    """Manage IoT simulation devices"""
    if not current_user.is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('auth.login'))

    # Get IoT device status from simulator
    from app.iot.iot_simulator import get_device_status
    devices = get_device_status()

    return render_template('admin/iot_devices.html', devices=devices)


@bp.route('/iot/device/<device_id>/control', methods=['POST'])
@login_required
def control_iot_device(device_id):
    """Control IoT device (start/stop/configure)"""
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403

    action = request.json.get('action')

    # Forward to IoT simulator
    from app.iot.iot_simulator import control_device
    result = control_device(device_id, action)

    # Log action
    log = AuditLog(
        user_id=current_user.id,
        action='IOT_DEVICE_CONTROL',
        ip_address=request.remote_addr,
        details=f'Device {device_id}: {action}'
    )
    db.session.add(log)
    db.session.commit()

    return jsonify(result)

