"""Lane-level signal programs for one SUMO traffic light.

A plan is a list of stages. A stage is (name, green seconds, lanes), where lanes
is a list of (incoming edge, lane indices); None means every lane of that edge.
Lanes that are green in two consecutive stages stay green through the yellow and
all-red between them.

`uturn_exits` maps the incoming edge of each upstream U-turn (made at the node
before the junction, but controlled by this light) to the junction exit that
feeds the road it turns into, e.g. {"A2_in1": "A2_out"}.

Used by road_builder when it writes the network and by
scripts/compare_signal_plans.py when it compares plans on an existing network.
"""
import itertools
import xml.etree.ElementTree as ET


def junction_links(net, tls):
    """Signal links of the light: index -> connection attributes, with the full path through the junction."""
    shape = {lane.get("id"): [tuple(map(float, p.split(","))) for p in lane.get("shape").split()]
             for lane in net.iter("lane")}
    following = {f'{c.get("from")}_{c.get("fromLane")}': c.get("via") for c in net.findall("connection")
                 if c.get("from").startswith(":") and c.get("via")}
    links = {}
    for c in net.findall("connection"):
        if c.get("tl") != tls:
            continue
        path, lane = [], c.get("via")
        while lane:
            path += shape[lane]
            lane = following.get(lane)
        links[int(c.get("linkIndex"))] = dict(c.attrib, path=path)
    return links


def stage_links(links, lanes):
    """Link indices that a stage turns green."""
    chosen = set()
    for edge, indices in lanes:
        found = {i for i, c in links.items() if c["from"] == edge
                 and (indices is None or int(c["fromLane"]) in indices)}
        if not found:
            raise ValueError(f"No signal link for {edge} lanes {indices}")
        chosen |= found
    return chosen


def green_state(links, green, uturn_exits):
    """'G' for protected green; 'g' (yield) for U-turns and a second stream into the same lane."""
    state, taken = ["r"] * (max(links) + 1), set()
    for i in sorted(green):
        c = links[i]
        target = (c["to"], c["toLane"])
        yields = c["dir"] == "t" or c["from"] in uturn_exits or target in taken
        state[i] = "g" if yields else "G"
        if not yields:
            taken.add(target)
    return state


def program(links, stages, yellow, all_red, uturn_exits, tls, program_id="0"):
    """SUMO tlLogic for the plan (green, yellow and all-red per stage) and the green links of each stage."""
    sets = [stage_links(links, lanes) for _, _, lanes in stages]
    logic = ET.Element("tlLogic", id=tls, type="static", programID=program_id, offset="0")
    for k, (name, seconds, _) in enumerate(stages):
        state = green_state(links, sets[k], uturn_exits)
        ending = sets[k] - sets[(k + 1) % len(sets)]
        ET.SubElement(logic, "phase", duration=f"{seconds:g}", state="".join(state), name=f"{name}_green")
        ET.SubElement(logic, "phase", duration=f"{yellow:g}",
                      state="".join("y" if i in ending else s for i, s in enumerate(state)),
                      name=f"{name}_yellow")
        ET.SubElement(logic, "phase", duration=f"{all_red:g}",
                      state="".join("r" if i in ending else s for i, s in enumerate(state)),
                      name=f"{name}_clearance")
    return logic, sets


def _cross(p, q):
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return any(side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0
               for a, b in zip(p[:-1], p[1:]) for c, d in zip(q[:-1], q[1:]))


def conflicts(links, green, uturn_exits):
    """Pairs of greens that must not share a stage.

    Two protected greens from different edges whose paths cross or end in the
    same lane, and an upstream U-turn together with a stream into its exit.
    """
    state = green_state(links, green, uturn_exits)
    found = []
    for i, j in itertools.combinations(sorted(green), 2):
        a, b = links[i], links[j]
        if a["from"] == b["from"] or "g" in (state[i], state[j]):
            continue
        if _cross(a["path"], b["path"]) or (a["to"], a["toLane"]) == (b["to"], b["toLane"]):
            found.append((i, j))
    for i in sorted(green):
        exit_edge = uturn_exits.get(links[i]["from"])
        found += [(i, j) for j in sorted(green) if exit_edge and links[j]["to"] == exit_edge]
    return found


def check(links, stages, sets, uturn_exits):
    """Raise if a stage holds conflicting greens or a link is never green."""
    for (name, _, _), green in zip(stages, sets):
        for i, j in conflicts(links, green, uturn_exits):
            a, b = links[i], links[j]
            raise ValueError(f"Signal stage {name}: {a['from']}_{a['fromLane']}>{a['to']} "
                             f"conflicts with {b['from']}_{b['fromLane']}>{b['to']}")
    for i in sorted(set(links) - set().union(*sets)):
        c = links[i]
        raise ValueError(f"Signal link {c['from']}_{c['fromLane']}>{c['to']} is never green")
