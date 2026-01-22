from __future__ import annotations

from functools import wraps
from typing import Callable, TypeVar, cast

from flask import redirect, session, url_for, request, jsonify
from sqlalchemy.orm import Session

from .models import User

F = TypeVar("F", bound=Callable[..., object])


def get_current_user(db: Session) -> User | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.get(User, int(user_id))


def login_required(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required."}), 401
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return cast(F, wrapper)
