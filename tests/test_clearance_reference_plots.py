import unittest
from unittest.mock import patch
from scripts.evaluate_quick_phase1 import summarize, _plot_clearances
from scripts.evaluate import plt


class ClearanceReferenceTests(unittest.TestCase):
    def test_stats_do_not_mix_references_and_keep_zero(self):
        rows=[dict(red_wait_clearance_m=value,red_wait_reference=ref,
                   first_signal_stop_clearance_m=None,green_restart_observed=False,
                   green_restart_delay_s=None)
              for value,ref in [(0,'stop_line'),(2,'stop_line'),(10,'queued_vehicle'),(20,'queued_vehicle'),(None,'queued_vehicle')]]
        with patch('scripts.evaluate_quick_phase1._summarize',return_value={}):
            result=summarize(rows)
        split=result['red_wait_clearance_by_reference']
        self.assertEqual(result['red_wait_clearance']['n'],4)
        self.assertEqual(split['stop_line']['n'],2)
        self.assertEqual(split['stop_line']['median_m'],1)
        self.assertEqual(split['queued_vehicle']['median_m'],15)
        self.assertEqual(split['queued_vehicle']['n'],2)
        with patch('scripts.evaluate_quick_phase1._summarize',return_value={}):
            empty=summarize(rows[:2])['red_wait_clearance_by_reference']['queued_vehicle']
        self.assertEqual(empty['n'],0)
        self.assertIsNone(empty['median_m'])

    def test_separate_panels_handle_absent_leader_samples(self):
        from scripts.evaluate_quick_phase1 import _red_wait_stats
        split={r:_red_wait_stats([dict(red_wait_clearance_m=1,red_wait_reference='stop_line')],reference=r)
               for r in ['stop_line','queued_vehicle']}
        summary={'all':{'red_wait_clearance_by_reference':split},
                 'by_route':{'route':{'red_wait_clearance_by_reference':split}}}
        fig,axes=plt.subplots(1,4)
        try:
            _plot_clearances(axes[0],axes[1],['route'],summary,'stop_line')
            _plot_clearances(axes[2],axes[3],['route'],summary,'queued_vehicle')
            self.assertIn('Stop-line',axes[0].get_title())
            self.assertIn('Stopped-leader',axes[2].get_title())
            fig.canvas.draw()
        finally:plt.close(fig)

    def test_final_panels_accept_115_second_cycle_offsets(self):
        from scripts.evaluate_quick_phase1 import _plot_final_panels
        rows = [dict(route='route', signal_offset_s=tick * .5, status='arrival')
                for tick in range(2, 230, 5)]
        fig = plt.figure()
        try:
            with patch('scripts.evaluate_quick_phase1._plot_clearances'):
                _plot_final_panels(fig, fig.add_gridspec(1, 1)[0], ['route'], rows, {})
            self.assertEqual(fig.axes[1].images[0].get_array().shape, (1, 46))
            self.assertLessEqual(fig.axes[1].get_xlim()[1], 115)
            fig.canvas.draw()
        finally:
            plt.close(fig)
