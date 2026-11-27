"""Build the current road (env/road_config.py) once per test process into a temporary folder."""
import atexit
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile

from env import road_builder
from env.road_config import EGO, ROAD, TRAFFIC


@lru_cache(maxsize=1)
def road_files():
    """(highway.sumocfg, highway.net.xml) of the freshly built road."""
    folder = Path(tempfile.mkdtemp(prefix="road-test-"))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    cfg = Path(road_builder.build(ROAD, EGO, TRAFFIC, str(folder)))
    return cfg, cfg.with_name("highway.net.xml")
