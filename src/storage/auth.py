"""User authentication and management."""

import os
import uuid
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv
import jwt

from .db import supabase

load_dotenv()

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days


# Pydantic models
class UserSignup(BaseModel):
    """Model for user signup."""
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)


class UserLogin(BaseModel):
    """Model for user login."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Model for user response."""
    id: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    last_login_at: Optional[datetime]


class AuthResponse(BaseModel):
    """Model for authentication response."""
    user: UserResponse
    token: str
    expires_at: datetime


# Password hashing functions
def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = os.urandom(32)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + pwdhash.hex()


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a stored password against provided password."""
    try:
        salt = bytes.fromhex(stored_password[:64])
        stored_hash = stored_password[64:]
        pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return hmac.compare_digest(pwdhash.hex(), stored_hash)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


# JWT token functions
def create_access_token(user_id: str, email: str) -> tuple[str, datetime]:
    """Create JWT access token."""
    expires_at = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": expires_at,
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, expires_at


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT access token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        print("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {e}")
        return None


# User CRUD operations
def create_user(user_data: UserSignup) -> Optional[UserResponse]:
    """Create a new user."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    try:
        # Check if user already exists
        existing = supabase.table("users").select("id").eq("email", user_data.email).execute()
        if existing.data and len(existing.data) > 0:
            print(f"❌ User with email {user_data.email} already exists")
            return None
        
        # Hash password
        password_hash = hash_password(user_data.password)
        
        # Create user
        user = {
            "email": user_data.email,
            "password_hash": password_hash,
            "full_name": user_data.full_name,
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("users").insert(user).execute()
        
        if result.data and len(result.data) > 0:
            user_dict = result.data[0]
            print(f"✅ Created user: {user_dict['email']}")
            return UserResponse(**user_dict)
        else:
            print("❌ Failed to create user")
            return None
            
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return None


def authenticate_user(login_data: UserLogin) -> Optional[UserResponse]:
    """Authenticate user with email and password."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    try:
        # Get user by email
        result = supabase.table("users").select("*").eq("email", login_data.email).execute()
        
        if not result.data or len(result.data) == 0:
            print(f"❌ User not found: {login_data.email}")
            return None
        
        user_dict = result.data[0]
        
        # Check if user is active
        if not user_dict.get('is_active', True):
            print(f"❌ User account is disabled: {login_data.email}")
            return None
        
        # Verify password
        if not verify_password(user_dict['password_hash'], login_data.password):
            print(f"❌ Invalid password for user: {login_data.email}")
            return None
        
        # Update last login time
        supabase.table("users").update({
            "last_login_at": datetime.utcnow().isoformat()
        }).eq("id", user_dict['id']).execute()
        
        print(f"✅ Authenticated user: {user_dict['email']}")
        return UserResponse(**user_dict)
        
    except Exception as e:
        print(f"❌ Error authenticating user: {e}")
        return None


def get_user_by_id(user_id: str) -> Optional[UserResponse]:
    """Get user by ID."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    try:
        result = supabase.table("users").select("*").eq("id", user_id).execute()
        
        if result.data and len(result.data) > 0:
            return UserResponse(**result.data[0])
        return None
        
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return None


def get_user_by_email(email: str) -> Optional[UserResponse]:
    """Get user by email."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    try:
        result = supabase.table("users").select("*").eq("email", email).execute()
        
        if result.data and len(result.data) > 0:
            return UserResponse(**result.data[0])
        return None
        
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return None
