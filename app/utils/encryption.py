# ============================================
# FILE: app/utils/encryption.py
# ============================================
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from flask import current_app


# Generate a key from a password (in production, use environment variables)
def get_encryption_key():
    """Get or create encryption key from environment"""
    key = os.getenv('ENCRYPTION_KEY')
    if not key:
        # Generate a key for development (in production, always set in .env)
        key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        print(f"WARNING: No ENCRYPTION_KEY set. Generated: {key}")
        print("Add this to your .env file for persistence.")
    return key.encode() if isinstance(key, str) else key


def get_cipher():
    """Get Fernet cipher instance"""
    key = get_encryption_key()
    return Fernet(key)


def encrypt_data(data):
    """
    Encrypt data using AES-256
    Args:
        data: String to encrypt
    Returns:
        Encrypted bytes
    """
    if data is None:
        return None

    cipher = get_cipher()
    if isinstance(data, str):
        data = data.encode('utf-8')
    return cipher.encrypt(data)


def decrypt_data(encrypted_data):
    """
    Decrypt data using AES-256
    Args:
        encrypted_data: Bytes to decrypt
    Returns:
        Decrypted string
    """
    if encrypted_data is None:
        return None

    cipher = get_cipher()
    try:
        decrypted = cipher.decrypt(encrypted_data)
        return decrypted.decode('utf-8')
    except Exception as e:
        current_app.logger.error(f"Decryption error: {e}")
        return "[DECRYPTION ERROR]"


def hash_file(file_data):
    """Generate SHA-256 hash of file for integrity verification"""
    import hashlib
    sha256 = hashlib.sha256()
    sha256.update(file_data)
    return sha256.hexdigest()


def generate_secure_filename(original_filename):
    """Generate a secure random filename"""
    import uuid
    from pathlib import Path

    ext = Path(original_filename).suffix
    secure_name = f"{uuid.uuid4().hex}{ext}"
    return secure_name
