# ============================================
# FILE: app/routes/auth_routes.py
# ============================================
from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.database.models import User, Patient, Doctor, AuditLog
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
import pyotp
import qrcode
import io
import base64

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login with SSO support"""
    if current_user.is_authenticated:
        return redirect_based_on_role()

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        totp_code = request.form.get('totp_code')

        user = User.query.filter_by(email=email).first()

        # Check if account is locked
        if user and user.account_locked_until and user.account_locked_until > datetime.utcnow():
            flash(f'Account is locked until {user.account_locked_until.strftime("%Y-%m-%d %H:%M")}. Try again later.',
                  'danger')
            return render_template('auth/login.html')

        # Verify password
        if not user or not user.check_password(password):
            if user:
                user.login_attempts += 1
                if user.login_attempts >= 5:
                    user.account_locked_until = datetime.utcnow() + timedelta(minutes=15)
                    flash('Too many failed attempts. Account locked for 15 minutes.', 'danger')
                db.session.commit()

            # Log failed attempt
            log = AuditLog(
                user_id=user.id if user else None,
                action='LOGIN_FAILED',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                details=f'Failed login attempt for email: {email}'
            )
            db.session.add(log)
            db.session.commit()

            flash('Invalid email or password', 'danger')
            return render_template('auth/login.html')

        # Check 2FA if enabled
        if user.two_factor_enabled:
            if not totp_code:
                session['pre_auth_user_id'] = user.id
                return render_template('auth/two_factor.html', email=email)

            totp = pyotp.TOTP(user.two_factor_secret)
            if not totp.verify(totp_code):
                flash('Invalid 2FA code', 'danger')
                return render_template('auth/two_factor.html', email=email)

        # Successful login
        user.login_attempts = 0
        user.last_login = datetime.utcnow()
        user.last_login_ip = request.remote_addr
        db.session.commit()

        login_user(user, remember=remember)

        # Log successful login
        log = AuditLog(
            user_id=user.id,
            action='LOGIN_SUCCESS',
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string,
            details=f'Successful login'
        )
        db.session.add(log)
        db.session.commit()

        flash('Logged in successfully!', 'success')
        return redirect_based_on_role()

    return render_template('auth/login.html')


@bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect_based_on_role()

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'patient')

        # Validation
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('auth/register.html')

        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('auth/register.html')

        if User.query.filter_by(username=username).first():
            flash('Username already taken', 'danger')
            return render_template('auth/register.html')

        # Create user
        user = User(
            username=username,
            email=email,
            role=role
        )
        user.set_password(password)

        db.session.add(user)
        db.session.flush()  # Get user ID

        # Create profile based on role
        if role == 'patient':
            patient = Patient(user_id=user.id)
            db.session.add(patient)
        elif role == 'doctor':
            doctor = Doctor(
                user_id=user.id,
                specialization=request.form.get('specialization', ''),
                license_number=request.form.get('license_number', '')
            )
            db.session.add(doctor)

        db.session.commit()

        # Log registration
        log = AuditLog(
            user_id=user.id,
            action='USER_REGISTERED',
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string,
            details=f'New {role} registration'
        )
        db.session.add(log)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')


@bp.route('/sso/<provider>')
def sso_login(provider):
    """Handle SSO login (simplified - in production use AuthLib)"""
    # This is a simplified version. In production, implement proper OAuth2 flow
    flash(f'{provider} SSO login - To be implemented with proper OAuth2', 'info')
    return redirect(url_for('auth.login'))


@bp.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    # Log logout
    log = AuditLog(
        user_id=current_user.id,
        action='LOGOUT',
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string
    )
    db.session.add(log)
    db.session.commit()

    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@bp.route('/setup-2fa')
@login_required
def setup_2fa():
    """Setup two-factor authentication"""
    if current_user.two_factor_enabled:
        flash('2FA is already enabled', 'info')
        return redirect(url_for('profile.settings'))

    # Generate secret
    secret = pyotp.random_base32()
    session['2fa_secret'] = secret

    # Generate QR code
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(
        name=current_user.email,
        issuer_name="Brain Tumor Detection System"
    )

    # Create QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to base64 for display
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template('auth/setup_2fa.html',
                           secret=secret,
                           qr_code=img_str)


@bp.route('/enable-2fa', methods=['POST'])
@login_required
def enable_2fa():
    """Enable 2FA after verification"""
    secret = session.get('2fa_secret')
    code = request.form.get('code')

    if not secret or not code:
        flash('Invalid request', 'danger')
        return redirect(url_for('auth.setup_2fa'))

    totp = pyotp.TOTP(secret)
    if totp.verify(code):
        current_user.two_factor_enabled = True
        current_user.two_factor_secret = secret
        db.session.commit()

        # Log 2FA enabled
        log = AuditLog(
            user_id=current_user.id,
            action='2FA_ENABLED',
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string
        )
        db.session.add(log)
        db.session.commit()

        flash('2FA enabled successfully!', 'success')
        session.pop('2fa_secret', None)
    else:
        flash('Invalid verification code', 'danger')

    return redirect(url_for('profile.settings'))


def redirect_based_on_role():
    """Redirect user based on their role"""
    if current_user.is_admin():
        return redirect(url_for('admin.dashboard'))
    elif current_user.is_doctor():
        return redirect(url_for('doctor.dashboard'))
    else:
        return redirect(url_for('patient.dashboard'))
