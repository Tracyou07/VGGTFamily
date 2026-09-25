import unittest
from unittest.mock import patch
import torch
from tests.ours_v7.test_model import ModelTest
from vggt.v7.model import WindowReconstructor

class DenseChunkTest(ModelTest):
    def test_chunked_outputs_tail_camera_and_devices(self):
        w=WindowReconstructor(self.m)
        kw=dict(window_size=5,overlap=1)
        reference=w(self.images,list("abcde"),**kw)["predictions"][0]
        for chunk in (1,2,4,8,16):
            with patch.object(self.m.camera_head,"forward",wraps=self.m.camera_head.forward) as camera:
                actual=w(self.images,list("abcde"),dense_head_frame_chunk=chunk,**kw)["predictions"][0]
                self.assertEqual(camera.call_count,1)
                self.assertEqual(camera.call_args.args[0][0].shape[1],5)
            self.assertEqual(actual["frame_ids"],list("abcde"))
            self.assertEqual(set(actual),set(reference))
            for key,value in reference.items():
                if key=="frame_ids":continue
                self.assertEqual(actual[key].device.type,"cpu")
                self.assertEqual(actual[key].dtype,torch.float32)
                self.assertFalse(actual[key].requires_grad)
                torch.testing.assert_close(actual[key],value,atol=2e-5,rtol=2e-5)
            for key in ("pose_encoding","c2w","intrinsics"):
                self.assertTrue(torch.equal(actual[key],reference[key]))
    def test_shared_features_live_through_both_heads_then_release(self):
        import weakref
        import gc
        import vggt.v7.model as module
        original=module.aggregate_windows
        refs=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            refs.extend(weakref.ref(x) for cache in result[0] for x in cache if x is not None)
            return result
        calls=[]
        def point_input(head,args):
            calls.append(1)
            self.assertTrue(all(ref() is not None for ref in refs))
        handle=self.m.point_head.register_forward_pre_hook(point_input)
        try:
            with patch.object(module,"aggregate_windows",new=capture):
                WindowReconstructor(self.m)(self.images,list("abcde"),window_size=5,
                                           overlap=1,dense_head_frame_chunk=2)
        finally:handle.remove()
        self.assertEqual(len(calls),3)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_invalid_chunk(self):
        with self.assertRaises(ValueError):
            WindowReconstructor(self.m)(self.images,list("abcde"),dense_head_frame_chunk=0)
if __name__=="__main__":unittest.main()
