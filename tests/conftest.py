"""Shared pytest fixtures. Makes ``src/`` importable without installing."""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rumen_twin.cow import CowPopulation          # noqa: E402
from rumen_twin.environment import RumenTwinEnv   # noqa: E402
from rumen_twin.utils import load_config, load_seeds  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def seeds():
    return load_seeds()


@pytest.fixture(scope="session")
def population(cfg, seeds):
    return CowPopulation(config=cfg, seeds=seeds)


@pytest.fixture
def env(cfg, seeds, population):
    return RumenTwinEnv(config=cfg, seeds=seeds, population=population, split="test",
                        scenarios=list(cfg["evaluation_scenarios"]))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
