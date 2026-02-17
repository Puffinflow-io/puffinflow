"""SQLAlchemy ORM models for PuffinFlow Studio."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=_new_id)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    settings_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    workflows = relationship(
        "Workflow", back_populates="project", cascade="all, delete-orphan"
    )
    eval_suites = relationship(
        "EvalSuite", back_populates="project", cascade="all, delete-orphan"
    )


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(String, primary_key=True, default=_new_id)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    yaml_content = Column(Text, default="")
    generated_python = Column(Text, default="")
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    project = relationship("Project", back_populates="workflows")
    deployments = relationship(
        "Deployment", back_populates="workflow", cascade="all, delete-orphan"
    )


class EvalSuite(Base):
    __tablename__ = "eval_suites"

    id = Column(String, primary_key=True, default=_new_id)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    scoring_config = Column(Text, default="{}")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    project = relationship("Project", back_populates="eval_suites")
    cases = relationship(
        "EvalCase", back_populates="suite", cascade="all, delete-orphan"
    )
    runs = relationship("EvalRun", back_populates="suite", cascade="all, delete-orphan")


class EvalCase(Base):
    __tablename__ = "eval_cases"

    id = Column(String, primary_key=True, default=_new_id)
    suite_id = Column(String, ForeignKey("eval_suites.id"), nullable=False)
    name = Column(String(255), nullable=False)
    input_data = Column(Text, default="{}")
    expected_output = Column(Text, default="{}")
    tags_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=_utcnow)

    suite = relationship("EvalSuite", back_populates="cases")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True, default=_new_id)
    suite_id = Column(String, ForeignKey("eval_suites.id"), nullable=False)
    workflow_id = Column(String, ForeignKey("workflows.id"), nullable=False)
    status = Column(String(50), default="pending")
    summary_scores = Column(Text, default="{}")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    suite = relationship("EvalSuite", back_populates="runs")
    workflow = relationship("Workflow")
    results = relationship(
        "EvalResult", back_populates="run", cascade="all, delete-orphan"
    )


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(String, primary_key=True, default=_new_id)
    run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    case_id = Column(String, ForeignKey("eval_cases.id"), nullable=False)
    actual_output = Column(Text, default="{}")
    scores_json = Column(Text, default="{}")
    latency_ms = Column(Float, default=0.0)
    passed = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)

    run = relationship("EvalRun", back_populates="results")
    case = relationship("EvalCase")


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(String, primary_key=True, default=_new_id)
    workflow_id = Column(String, ForeignKey("workflows.id"), nullable=False)
    target = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    url = Column(String(500), default="")
    logs = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    workflow = relationship("Workflow", back_populates="deployments")
