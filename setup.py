# setup.py
import subprocess
import sys
import os


def install_requirements():
    """Install all required packages"""
    packages = [
        'cryptography==41.0.3',
        'tensorflow==2.13.0',
        'scikit-learn==1.3.0',
        'opencv-python==4.8.1.78',
        'pillow==10.0.1',
        'numpy==1.24.3',
        'pandas==2.0.3',
        'matplotlib==3.7.2',
        'seaborn==0.12.2',
        'flask==2.3.3',
        'flask-sqlalchemy==3.1.1',
        'flask-login==0.6.2',
        'flask-wtf==1.1.1',
        'flask-migrate==4.0.5',
        'flask-cors==4.0.0',
        'flask-mail==0.9.1',
        'werkzeug==2.3.7',
        'sqlalchemy==2.0.20',
        'alembic==1.11.1',
        'bcrypt==4.0.1',
        'pyjwt==2.8.0',
        'passlib==1.7.4',
        'itsdangerous==2.1.2',
        'pyotp==2.8.0',
        'qrcode==7.4.2',
        'reportlab==4.0.4',
        'python-dotenv==1.0.0',
        'email-validator==2.0.0',
        'requests==2.31.0',
        'tqdm==4.65.0',
        'psutil==5.9.5'
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("All packages installed successfully!")


if __name__ == "__main__":
    install_requirements()