import unittest
import numpy as np
import torch
from scripts.train_bc_shards import lane_class, export_policy, initialize_from_raw_policy
from utils.networks import HybridActorCritic


class BCActorTests(unittest.TestCase):
    def test_continuation_preserves_raw_actor_with_new_statistics(self):
        torch.manual_seed(7)
        original=HybridActorCritic(43,(256,256),'relu').eval()
        continued=HybridActorCritic(43,(256,256),'relu').eval()
        mean=torch.randn(43);std=torch.rand(43)+.1;raw=torch.randn(64,43)
        initialize_from_raw_policy(continued,original.state_dict(),mean,std)
        with torch.no_grad():
            expected=original(raw);actual=continued((raw-mean)/std)
        for i in [0,2,3]:
            torch.testing.assert_close(actual[i],expected[i],rtol=1e-4,atol=1e-5)

    def test_class_order_matches_ppo_commands(self):
        labels=lane_class(np.array([[0,1],[0,0],[0,-1]],np.float32))
        np.testing.assert_array_equal(labels,[0,1,2])
        with self.assertRaises(ValueError):lane_class(np.array([[0,.2]]))

    def test_folded_actor_preserves_predictions_on_raw_observations(self):
        torch.manual_seed(0)
        model=HybridActorCritic(43,(256,256),'relu').eval()
        mean=torch.randn(43);std=torch.rand(43)+.1;raw=torch.randn(64,43)
        exported=HybridActorCritic(43,(256,256),'relu').eval()
        exported.load_state_dict(export_policy(model,mean,std))
        with torch.no_grad():
            a=model((raw-mean)/std);b=exported(raw)
        for i in [0,2,3]:torch.testing.assert_close(a[i],b[i],rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(model.det_action((raw-mean)/std),exported.det_action(raw),rtol=1e-4,atol=1e-5)
