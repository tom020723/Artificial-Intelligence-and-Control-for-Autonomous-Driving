import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from scripts.evaluate_quick_bc import prepare_context, plot_bc_training, configure_ego_collision_gap
from scripts.evaluate import plt


class BCQuickEvaluationTests(unittest.TestCase):
    def test_collision_override_preserves_observation_gap_and_background_types(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'test.sumocfg'
            config.write_text('<configuration><input><route-files value="types.xml"/></input></configuration>')
            types = root / 'types.xml'
            types.write_text('<routes><vType id="ego_type"/><vType id="car_krauss" minGap="2.5"/></routes>')
            result = configure_ego_collision_gap(config, .2)
            nodes = {node.get('id'): node for node in ET.parse(types).getroot()}
            self.assertIsNone(nodes['ego_type'].get('minGap'))
            self.assertAlmostEqual(float(nodes['ego_type'].get('collisionMinGapFactor')), .08)
            self.assertEqual(nodes['car_krauss'].attrib, dict(id='car_krauss', minGap='2.5'))
            self.assertEqual(result['ego_collision_gap_m'], .2)
            self.assertEqual(result['ego_min_gap_m'], 2.5)
            for invalid in [-1, float('nan'), float('inf')]:
                with self.assertRaises(ValueError):
                    configure_ego_collision_gap(config, invalid)

    def test_context_restores_collection_signals_and_background_without_editing_run(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();data=root/'data';data.mkdir()
            (run/'model.pt').write_bytes(b'fixture')
            (run/'road_config.py').write_text("ROUTE_ONLY_EXPERIMENT=True\nSIGNAL_EXPERIMENT=False\nTRAFFIC={}\nNOHYEONG={'ring_segments':[{'vph':0}], 'loop_routes':[]}\n_RING_BACKGROUND_VPH=(80,)\n")
            (run/'mdp_config.py').write_text("OBSERVATION={}\nSIMULATION={}\nACTION={'lane_change_safety':{'enabled':True,'spawn_protect':60}}\n")
            (run/'train.py').write_text('HPARAMS={}\n')
            (run/'config.json').write_text(json.dumps(dict(algorithm='supervised_bc_on_ppo_hybrid_actor',
                    data=str(data/'dataset.npz'),hidden_sizes=[256,256],activation='relu')))
            (data/'config.json').write_text(json.dumps(dict(observation=dict(layout='bc43'),
                    traffic=dict(vehs_per_hour=800,background_enabled=True,lc_speed_gain=1),
                    simulation=dict(step_length=.5))))
            saved,curriculum,_=prepare_context(run,root/'context')
            namespace={}
            exec((root/'context/road_config.py').read_text(),namespace)
            self.assertFalse(namespace['ROUTE_ONLY_EXPERIMENT'])
            self.assertTrue(namespace['SIGNAL_EXPERIMENT'])
            self.assertTrue(namespace['TRAFFIC']['background_enabled'])
            self.assertFalse(namespace['NOHYEONG']['all_signals_green'])
            self.assertEqual(namespace['NOHYEONG']['ring_segments'][0]['vph'],80)
            self.assertIn('ROUTE_ONLY_EXPERIMENT=True',(run/'road_config.py').read_text())
            mdp={}
            exec((root/'context/mdp_config.py').read_text(),mdp)
            self.assertEqual(mdp['ACTION']['lane_change_safety']['spawn_protect'],0)
            self.assertFalse(mdp['ACTION']['lane_change_safety']['enabled'])
            self.assertFalse(mdp['ACTION']['lane_change_safety']['spawn_protect_only'])
            self.assertIn("'spawn_protect':60",(run/'mdp_config.py').read_text())
            self.assertTrue(saved['evaluation_only'])
            self.assertTrue(curriculum.total_vph)
            self.assertEqual(curriculum.ring_reference_vph,800)
            meta=json.loads((data/'config.json').read_text())
            meta['expert_control_timing']={'spawn_protect':30,'block_internal_at_step_start':True}
            (data/'config.json').write_text(json.dumps(meta))
            updated,_,_=prepare_context(run,root/'aligned_context')
            aligned={}
            exec((root/'aligned_context/mdp_config.py').read_text(),aligned)
            # The expert's entry restriction is recorded but never applied to the BC ego.
            self.assertEqual(aligned['ACTION']['lane_change_safety']['spawn_protect'],0)
            self.assertFalse(aligned['ACTION']['lane_change_safety']['enabled'])
            self.assertEqual(updated['bc_spawn_protect_m'],0)
            self.assertEqual(updated['expert_spawn_protect_m'],30)

    def test_training_panels_use_epoch_and_bc_losses(self):
        with TemporaryDirectory() as directory:
            run=Path(directory)
            (run/'training_log.csv').write_text('epoch,train_loss,val_loss,val_accel_mse,val_lane_ce,val_lane_macro_f1,val_left_precision,val_right_precision,val_left_recall,val_right_recall\n1,.8,.9,.1,.8,.2,.3,.4,.5,.6\n2,.4,.5,.1,.4,.4,.5,.6,.7,.8\n')
            (run/'result.json').write_text('{"best_epoch":2}')
            fig=plt.figure();grid=fig.add_gridspec(4,2)
            try:
                plot_bc_training(fig,grid,run)
                self.assertEqual(len(fig.axes),6)
                self.assertTrue(all(ax.get_xlabel()=='Epoch' for ax in fig.axes))
                self.assertEqual(list(fig.axes[0].lines[0].get_xdata()),[1,2])
                self.assertEqual(list(fig.axes[0].lines[0].get_ydata()),[.8,.4])
                self.assertTrue(all('PPO' not in ax.get_title() for ax in fig.axes))
            finally:
                plt.close(fig)
