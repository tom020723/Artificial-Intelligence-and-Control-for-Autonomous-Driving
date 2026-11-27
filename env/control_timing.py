"""Shared lane-change timing constraints for demonstrations and ego execution."""

def lane_change_timing_allowed(edge, lane_position, route_index, spawn_protect):
    return (not edge.startswith(':')
            and not (route_index == 0 and lane_position < spawn_protect))
