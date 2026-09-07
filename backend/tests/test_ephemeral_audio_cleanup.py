"""Tests for deleting agent audio without deleting its history entry."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, Generation, VoiceProfile
from backend.models import HistoryQuery
from backend.services import history


@pytest.fixture
def db(tmp_path):
    original_data_dir = config.get_data_dir()
    config.set_data_dir(tmp_path)

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    session.add(VoiceProfile(id="profile-1", name="Test Profile"))
    session.commit()

    try:
        yield session
    finally:
        session.close()
        config.set_data_dir(original_data_dir)


@pytest.mark.asyncio
async def test_delete_generation_audio_keeps_history_row(db):
    audio_path = Path("generations/agent.wav")
    audio_file = config.get_data_dir() / audio_path
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"audio")

    db.add(
        Generation(
            id="generation-1",
            profile_id="profile-1",
            text="Temporary agent response",
            audio_path=str(audio_path),
            keep_audio=False,
        )
    )
    db.commit()

    assert await history.delete_generation_audio("generation-1", db) is True
    assert not audio_file.exists()

    row = db.query(Generation).filter_by(id="generation-1").one()
    assert row.audio_path is None

    listed = await history.list_generations(HistoryQuery(limit=10), db)
    assert listed.total == 0
    assert listed.items == []


@pytest.mark.asyncio
async def test_delete_generation_audio_reports_missing_generation(db):
    assert await history.delete_generation_audio("missing", db) is False
