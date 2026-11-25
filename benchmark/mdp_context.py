# Benchmark observation: the bc49 layout with split-guidance lateral distances
# (scale 50 m). The BC collection adds its own fields on top of this
# (scripts/measure_calibrated_idm_coverage.py), giving the final 52-dim input.
OBSERVATION = {'neighbor_distance_mode': 'bumper', 'layout': 'bc49', 'signal_distance_scale': 200.0, 'guidance_distance_scale': 650.0, 'connectivity_half_width': 2, 'lookahead_connectivity_enabled': False, 'destination_enabled': False, 'destination_arms': ['A1', 'A2', 'A3', 'A4', 'A5'], 'signal_enabled': True, 'signal_visibility': 100.0, 'signal_count': 1, 'visibility': 50.0, 'density_veh_length': 7.5, 'ring_visibility': 50.0, 'ring_distance_scale': 50.0, 'guidance_lateral_scale': 50.0}
