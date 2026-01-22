from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _json_type():
    return JSONB().with_variant(JSON(), "sqlite")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    resumes = relationship("Resume", back_populates="user", cascade="all, delete-orphan")
    company_profiles = relationship("CompanyProfile", back_populates="user", cascade="all, delete-orphan")
    interviews = relationship("Interview", back_populates="user", cascade="all, delete-orphan")


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="resumes")


class CompanyProfile(Base):
    __tablename__ = "company_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    role: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    details: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="company_profiles")


class Interview(Base):
    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    resume_id: Mapped[int | None] = mapped_column(ForeignKey("resumes.id"), nullable=True)
    company_profile_id: Mapped[int | None] = mapped_column(ForeignKey("company_profiles.id"), nullable=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="full")
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="python")
    model: Mapped[str] = mapped_column(String(64), nullable=False, default="gemini-2.5-flash-lite")

    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    current_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    phase_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    phase_turn_start_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_speech_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_code_change_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    candidate_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    problem_presented: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    current_code: Mapped[str] = mapped_column(Text, nullable=False, default="")
    conversation: Mapped[list] = mapped_column(MutableList.as_mutable(_json_type()), nullable=False, default=list)
    code_snapshots: Mapped[list] = mapped_column(MutableList.as_mutable(_json_type()), nullable=False, default=list)

    company_context: Mapped[dict] = mapped_column(MutableDict.as_mutable(_json_type()), nullable=False, default=dict)
    coding_question: Mapped[dict] = mapped_column(MutableDict.as_mutable(_json_type()), nullable=False, default=dict)
    ood_question: Mapped[dict] = mapped_column(MutableDict.as_mutable(_json_type()), nullable=False, default=dict)
    code_evaluation: Mapped[dict] = mapped_column(MutableDict.as_mutable(_json_type()), nullable=False, default=dict)
    coding_summary: Mapped[dict] = mapped_column(MutableDict.as_mutable(_json_type()), nullable=False, default=dict)
    report: Mapped[dict] = mapped_column(MutableDict.as_mutable(_json_type()), nullable=False, default=dict)

    user = relationship("User", back_populates="interviews")
    resume = relationship("Resume")
    company_profile = relationship("CompanyProfile")
