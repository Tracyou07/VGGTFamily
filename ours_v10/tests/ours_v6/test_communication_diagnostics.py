import unittest
from unittest.mock import patch
import torch
from torch import nn
from vggt.layers.block import Block
from vggt.v6.attention import global_step

torch.set_num_threads(1)

class CommunicationDiagnosticTest(unittest.TestCase):
    def setUp(self):
        block = Block(8, 2, qkv_bias=False).double().eval().requires_grad_(False)
        block.norm1 = nn.Identity()
        block.norm2 = nn.Identity()
        with torch.no_grad():
            block.attn.qkv.weight.zero_()
            block.attn.qkv.weight[16:] = torch.eye(8, dtype=torch.float64)
            block.attn.proj.weight.copy_(torch.eye(8, dtype=torch.float64))
            block.attn.proj.bias.zero_()
            for p in block.mlp.parameters(): p.zero_()
        self.block = block
        self.states = [torch.tensor([[[1.]*8,[1.]*8,[0.]*8,[0.]*8]], dtype=torch.float64),
                       torch.tensor([[[10.]*8,[10.]*8,[0.]*8,[0.]*8]], dtype=torch.float64)]

    def run_mode(self, mode, records=None):
        return global_step(self.block, self.states, [None,None], 4, mode, 1, 1,
                           diagnostics=records, layer=0)

    def test_value_changes_only_allowed_query_types(self):
        baseline = self.run_mode('independent')
        camera = self.run_mode('camera_exchange')
        both = self.run_mode('camera_register_exchange')
        self.assertGreater(float((camera[0][0,0]-baseline[0][0,0]).abs().max()), 0.1)
        self.assertTrue(torch.equal(camera[0][0,1:], baseline[0][0,1:]))
        self.assertGreater(float((both[0][0,0]-camera[0][0,0]).abs().max()), 0.1)
        self.assertGreater(float((both[0][0,1]-camera[0][0,1]).abs().max()), 0.1)
        self.assertTrue(torch.equal(both[0][0,2:], camera[0][0,2:]))

    def test_probe_does_not_change_results_and_zero_bank_changes_specials(self):
        for mode in ('independent','camera_exchange','camera_register_exchange'):
            reference=self.run_mode(mode)
            rows=[]
            probed=self.run_mode(mode,rows)
            for a,b in zip(reference,probed): self.assertTrue(torch.equal(a,b))
            if mode=='independent': continue
            from vggt.v6 import attention as module
            original=module.communication_bank
            def zeroed(*args,**kwargs):
                k,v=original(*args,**kwargs)
                return torch.zeros_like(k),torch.zeros_like(v)
            with patch.object(module,'communication_bank',side_effect=zeroed):
                ablated=self.run_mode(mode)
            self.assertGreater(float((reference[0][0,0]-ablated[0][0,0]).abs().max()),0.1)
            if mode=='camera_register_exchange':
                self.assertGreater(float((reference[0][0,1]-ablated[0][0,1]).abs().max()),0.1)
            self.assertTrue(torch.equal(reference[0][0,2:],ablated[0][0,2:]))

    def test_diagnostics_count_and_mass(self):
        for mode, count in [('independent',0),('camera_exchange',1),('camera_register_exchange',2)]:
            records=[]
            self.run_mode(mode, records)
            self.assertEqual(len(records),2)
            for row in records:
                self.assertEqual(row['mode'],mode)
                self.assertEqual(row['layer'],0)
                self.assertEqual(row['window_count'],2)
                self.assertEqual(row['camera_count'],1)
                self.assertEqual(row['register_count'],1)
                self.assertEqual(row['remote_kv_tokens'],count)
                self.assertEqual(row['local_kv_tokens'],4)
                self.assertGreater(row['remote_attention_mass'],0 if count else -1)
                self.assertLessEqual(row['remote_attention_mass'],1)
                self.assertEqual(len(row['heads']),2)
                self.assertEqual(row['token_layout']['patch_start_idx'],2)
                self.assertTrue(row['bank_checksum'])
        no=[]; self.run_mode('independent',no)
        self.assertEqual(no[0]['communication_bank_bytes'],0)
        self.assertEqual(no[0]['remote_attention_mass'],0)

if __name__=='__main__': unittest.main()
