"""Explicit physical collision gaps, independent of car-following minGap, and SUMO collision options."""
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def configure_collision_gaps(config, *, ego_gap_m=None, background_gap_m=None):
    for gap in (ego_gap_m, background_gap_m):
        if gap is not None and (not math.isfinite(gap) or gap < 0):
            raise ValueError('Collision gaps must be finite and nonnegative')
    config = Path(config)
    inputs = ET.parse(config).getroot().find('input')
    settings = {}
    for filename in inputs.find('route-files').get('value').split(','):
        path = config.parent / filename.strip()
        tree = ET.parse(path)
        changed = False
        for node in tree.getroot().iter('vType'):
            name = node.get('id')
            gap = ego_gap_m if name == 'ego_type' else background_gap_m
            if gap is None:
                continue
            min_gap = float(node.get('minGap', '2.5'))
            if not math.isfinite(min_gap) or min_gap <= 0:
                raise ValueError('Vehicle minGap must be positive to scale collision detection')
            factor = gap / min_gap
            node.set('collisionMinGapFactor', repr(factor))
            settings[name] = dict(collision_gap_m=gap, min_gap_m=min_gap,
                                  collision_min_gap_factor=factor)
            changed = True
        if changed:
            tree.write(path, encoding='utf-8', xml_declaration=True)
    if ego_gap_m is not None and 'ego_type' not in settings:
        raise ValueError('Missing ego_type in collision configuration')
    return settings


JUNCTION_COLLISIONS_VARIABLE = "SUMO_CHECK_JUNCTIONS"


def junction_collisions():
    """Whether SUMO also detects collisions inside junctions (environment SUMO_CHECK_JUNCTIONS=1).

    Off by default, as when the final models were built. The setting is read
    from the environment so every SUMO a run starts, including spawned worker
    processes, uses the same rule.
    """
    value = os.environ.get(JUNCTION_COLLISIONS_VARIABLE, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{JUNCTION_COLLISIONS_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


def sumo_collision_options():
    """Extra SUMO collision options for every simulation start; empty by default."""
    return ["--collision.check-junctions", "true"] if junction_collisions() else []
