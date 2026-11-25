# BC evaluation context (not a PPO-trained run).
ROUTE_ONLY_EXPERIMENT = False
SIGNAL_EXPERIMENT = True
TRAFFIC = {'collision_gap_m': 0.2, 'right_turn_only_lane': True, 'startup_spread_seconds': 60, 'lc_speed_gain': 1.0, 'lc_keep_right': 0.0, 'max_speed': 16.0, 'vehs_per_hour': 800, 'depart_lane': 'random', 'keep_lane': False, 'depart_speed': 8.0, 'controllers': {'krauss': {'probability': 0.35, 'carFollowModel': 'Krauss', 'accel': 2.0, 'decel': 4.5, 'tau': 1.0, 'min_gap': 2.5, 'sigma': 0.3, 'max_speed': 14.0, 'speed_dev': 0.15, 'color': '1,1,0', 'keep_lane': False}, 'idm': {'probability': 0.3, 'carFollowModel': 'IDM', 'accel': 1.8, 'decel': 3.5, 'tau': 1.2, 'min_gap': 2.0, 'max_speed': 16.0, 'speed_dev': 0.1, 'color': '0,1,1', 'keep_lane': False}, 'eidm': {'probability': 0.2, 'carFollowModel': 'EIDM', 'accel': 2.0, 'decel': 4.0, 'tau': 1.1, 'min_gap': 2.0, 'max_speed': 16.5, 'speed_dev': 0.1, 'color': '1,0.6,0', 'keep_lane': False}, 'acc': {'probability': 0.15, 'carFollowModel': 'ACC', 'accel': 2.0, 'decel': 4.0, 'tau': 1.5, 'min_gap': 2.0, 'max_speed': 19.0, 'speed_dev': 0.05, 'color': '1,0,1', 'keep_lane': False}}, 'prefill': {'enabled': False, 'start_offset': 80, 'end_margin': 50, 'spacing': None, 'spacing_jitter': 0.45, 'lane_spacing_factor': [0.85, 1.0, 1.35]}, 'background_enabled': True, 'demand_profile': 'weighted_od'}
NOHYEONG['all_signals_green'] = False
NOHYEONG['warmup_seconds'] = 0
for demand, vph in zip(NOHYEONG['ring_segments'] + NOHYEONG['loop_routes'], _RING_BACKGROUND_VPH, strict=True):
    demand['vph'] = vph

EGO = dict(EGO, depart_lane='center')
