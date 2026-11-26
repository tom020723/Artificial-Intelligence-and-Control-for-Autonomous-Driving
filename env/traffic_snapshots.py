"""Continuously refreshed, per-demand traffic states from ego-free SUMO roads."""

from functools import partial
from itertools import count
import json
from multiprocessing import get_context
from pathlib import Path
import signal
import sqlite3
from tempfile import TemporaryDirectory
import traceback

import numpy as np


def _generate(config, directory, curriculum, reference_vph, levels, step, seed, refresh, ready, stop,
              capture_steps=None, adaptive=False, uniform_idm_vph=0):
    # Import inside the spawned process: the caller may already own a libsumo instance.
    from env.collision_detection import sumo_collision_options
    from env.parallel_sumo import _build_replicas, _replica_id, libsumo
    from env.parallel_traffic import TrafficSchedule

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    db = None
    try:
        directory = Path(directory)
        snapshot_config, _, lights, ends, flows, _, _ = _build_replicas(
            config, directory, len(levels), step, True, curriculum, reference_vph)
        libsumo.start(["sumo", "-c", snapshot_config, "--step-length", str(step),
                       "--seed", str(seed), "--collision.action", "remove",
                       "--no-warnings", "true", "--no-step-log", "true",
                       *curriculum.sumo_options(), *sumo_collision_options()])
        if adaptive:
            from env.adaptive_traffic_schedule import adaptive_schedule_factory
            factory = adaptive_schedule_factory(config, directory, flows, lights, uniform_idm_vph)
            schedules = [factory() for _ in levels]
        else:
            schedules = [TrafficSchedule(flows) for _ in levels]
        prefixes = [partial(_replica_id, i) for i in range(len(levels))]
        rngs = [np.random.default_rng(np.random.SeedSequence([seed, vph])) for vph in levels]
        for schedule, vph, rng in zip(schedules, levels, rngs, strict=True):
            schedule.reset(libsumo, 0., vph, rng, set())

        db = sqlite3.connect(directory / "states.sqlite")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE states (vph INTEGER, tick INTEGER, slot INTEGER, vehicles TEXT, "
                   "PRIMARY KEY (vph, tick, slot)) WITHOUT ROWID")
        warmup = int(np.ceil(curriculum.snapshot_warmup_seconds / step))
        capture_steps = set(capture_steps) if capture_steps is not None else None
        target_count = len(capture_steps) if capture_steps is not None else int(ends[-1])
        if capture_steps is not None:
            db.execute("CREATE TABLE signal_ticks (snapshot_tick INTEGER PRIMARY KEY, signal_tick INTEGER)")
        if adaptive:
            db.execute("CREATE TABLE adaptive_states (vph INTEGER, tick INTEGER, state TEXT, PRIMARY KEY (vph, tick))")
        cycle = int(ends[-1])
        tt = _replica_id(0, lights[0])
        v = libsumo.vehicle
        batch = {}
        generation = 0
        for step_index in count(1):
            if stop.is_set():
                break
            now = libsumo.simulation.getTime()
            for schedule, prefix, rng in zip(schedules, prefixes, rngs, strict=True):
                schedule.spawn_due(libsumo, now, prefix, rng)
            libsumo.simulationStep()
            active = set(v.getIDList())
            live = active | set(libsumo.simulation.getPendingVehicles())
            for schedule in schedules:
                schedule.ids.intersection_update(live)
            if step_index <= warmup:
                continue
            if capture_steps is not None and step_index not in capture_steps:
                continue
            now = libsumo.simulation.getTime()
            phase = libsumo.trafficlight.getPhase(tt)
            remaining = round((libsumo.trafficlight.getNextSwitch(tt) - now) / step)
            tick = int(ends[phase] - remaining) % cycle
            if capture_steps is not None:
                db.execute("INSERT INTO signal_ticks VALUES (?, ?)", (step_index, tick))
                tick = step_index
            for vph, schedule, prefix in zip(levels, schedules, prefixes, strict=True):
                vehicles = []
                for vid in sorted(schedule.ids & active):
                    lane = v.getLaneID(vid)
                    if not lane:  # SUMO may temporarily take a vehicle off the road.
                        continue
                    lane = (":" + lane.removeprefix(":" + prefix("")) if lane.startswith(":")
                            else lane.removeprefix(prefix("")))
                    vehicles.append((v.getRouteID(vid).removeprefix(prefix("")), v.getTypeID(vid),
                                     lane, v.getLanePosition(vid), v.getSpeed(vid),
                                     v.getSpeedFactor(vid), v.getLength(vid)))
                if adaptive:
                    db.execute("INSERT INTO adaptive_states VALUES (?, ?, ?)",
                               (vph, tick, json.dumps(schedule.checkpoint(now))))
                batch[vph, tick] = json.dumps(vehicles, separators=(",", ":"))
            if len(batch) == len(levels) * target_count:
                with db:
                    db.executemany("INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?)",
                                   ((vph, tick, generation % 3, vehicles)
                                    for (vph, tick), vehicles in batch.items()))
                batch.clear()
                if generation == 0:
                    ready.send(None)
                    ready.close()
                    if not refresh:
                        break
                generation += 1
            # Fill the first cycle quickly; refresh afterwards alongside training.
            if generation and stop.wait(0.02):
                break
    except Exception:
        if not ready.closed:
            ready.send(traceback.format_exc())
        raise
    finally:
        if db is not None:
            db.close()
        libsumo.close()
        ready.close()


class TrafficSnapshotBank:
    """Own the generator process; readers in rollout workers only need its path."""

    def __init__(self, config, curriculum, reference_vph, levels, step, seed, *, refresh,
                 capture_steps=None, adaptive=False, uniform_idm_vph=0):
        if capture_steps is not None:
            capture_steps = tuple(set(capture_steps))
            if refresh or not capture_steps or min(capture_steps) <= int(np.ceil(curriculum.snapshot_warmup_seconds / step)):
                raise ValueError("Fixed capture steps must follow warm-up and cannot be refreshed")
        if adaptive and capture_steps is None:
            raise ValueError("Adaptive snapshots require fixed capture steps")
        ctx = get_context("spawn")
        self.directory = TemporaryDirectory(prefix="sumo-traffic-states-")
        self.path = str(Path(self.directory.name) / "states.sqlite")
        self.stop = ctx.Event()
        self.refresh = refresh
        parent, child = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=_generate,
            args=(config, self.directory.name, curriculum, reference_vph, levels, step, seed, refresh,
                  child, self.stop, capture_steps, adaptive, uniform_idm_vph), daemon=True)
        try:
            self.process.start()
            child.close()
            if not parent.poll(120):
                raise RuntimeError("Traffic snapshot generator did not fill its first signal cycle")
            error = parent.recv()
            if error:
                raise RuntimeError(error)
        except BaseException:
            self.close()
            raise
        finally:
            parent.close()
            child.close()

    def check(self):
        if self.refresh and self.process.exitcode is not None:
            raise RuntimeError(f"Traffic snapshot generator exited ({self.process.exitcode})")

    def close(self):
        self.stop.set()
        if self.process.pid is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        self.process.close()
        self.directory.cleanup()


def pick_snapshot(db, vph, tick, rng, lane, position, clearance):
    """Prefer a state with room to insert the ego at its chosen departure."""
    rows = db.execute("SELECT vehicles FROM states WHERE vph=? AND tick=? ORDER BY slot",
                      (vph, tick)).fetchall()
    if not rows:
        raise RuntimeError(f"Missing traffic snapshot at {vph} vehicles/hour and signal tick {tick}")

    def free(vehicle):
        return vehicle[2] != lane or (vehicle[3] - vehicle[6] >= position + clearance
                                      or vehicle[3] <= position - clearance)

    for i in rng.permutation(len(rows)):
        vehicles = json.loads(rows[i][0])
        if all(map(free, vehicles)):
            return vehicles
    # Keep the sampled lane and signal, and omit only cars blocking ego insertion.
    return [vehicle for vehicle in vehicles if free(vehicle)]
