import unittest
import torch
from tests.ours_v6.test_scheduler import tiny
from vggt.v6.scheduler import aggregate_windows

class SchedulerDiagnosticTest(unittest.TestCase):
    def test_every_layer_window_and_register_writeback(self):
        ag=tiny()
        images=torch.rand(5,3,4,4,dtype=torch.float64)
        windows=[(0,4),(2,5)]
        for mode in ('independent','camera_exchange','camera_register_exchange'):
            records=[]
            aggregate_windows(ag,images,windows,mode,diagnostics=records,
                              frame_ids=[f'f{i}' for i in range(5)])
            self.assertEqual(len(records),ag.depth*len(windows))
            self.assertEqual({r['layer'] for r in records},set(range(ag.depth)))
            for row in records:
                self.assertEqual(row['window_range'],list(windows[row['window_index']]))
                self.assertEqual(len(row['frame_ids']),windows[row['window_index']][1]-windows[row['window_index']][0])
                self.assertGreater(row['register_input_norm'],0)
                self.assertGreater(row['register_output_norm'],0)
                if row['layer']<ag.depth-1:
                    self.assertTrue(row['register_writeback_preserved'])
                    self.assertEqual(row['register_output_checksum'],row['next_layer_register_input_checksum'])

if __name__=='__main__': unittest.main()
