import unittest
from unittest.mock import patch
import torch
from torch import nn
from vggt.layers.block import Block
from vggt.layers.attention import Attention
from vggt.v5.attention import project_qkv, camera_bank, exchange_attention, global_step

torch.set_num_threads(1)
class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.attn=Attention(8,2,qk_norm=True).double().eval()
        self.x=[torch.randn(1,6,8,dtype=torch.float64),torch.randn(1,9,8,dtype=torch.float64)]
        self.p=3
    def dense(self,attn,x):
        qs=[];ks=[];vs=[]; owners=[]; cameras=[]
        for w,t in enumerate(x):
            q,k,v=project_qkv(attn,t,None); qs.append(q);ks.append(k);vs.append(v)
            owners.extend([w]*t.shape[1]); cameras.extend(i%self.p==0 for i in range(t.shape[1]))
        q,k,v=[torch.cat(z,dim=2) for z in (qs,ks,vs)]
        owner=torch.tensor(owners); cam=torch.tensor(cameras)
        mask=(owner[:,None]==owner[None,:]) | (cam[:,None]&cam[None,:])
        y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        y=attn.proj_drop(attn.proj(y.transpose(1,2).reshape(1,-1,8)))
        return list(y.split([t.shape[1] for t in x],dim=1))
    def run_exchange(self,x,attn=None):
        a=attn or self.attn
        bank=[camera_bank(a,t,None,self.p) for t in x]
        return [exchange_attention(a,t,None,self.p,bank,i) for i,t in enumerate(x)]
    def test_dense_equivalence_and_no_mask(self):
        expected=self.dense(self.attn,self.x); calls=[]
        real=torch.nn.functional.scaled_dot_product_attention
        def trace(q,k,v,**kw):
            self.assertIsNone(kw.get('attn_mask')); calls.append((q.shape[2],k.shape[2])); return real(q,k,v,**kw)
        with patch('torch.nn.functional.scaled_dot_product_attention',trace): actual=self.run_exchange(self.x)
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)
        self.assertNotIn((15,15),calls)
        self.assertEqual(sum(q*k for q,k in calls),6*4+2*9+9*6+3*11)
    def controlled(self):
        a=Attention(8,2,qkv_bias=False).double().eval()
        with torch.no_grad():
            a.qkv.weight.zero_();a.qkv.weight[16:]=torch.eye(8,dtype=torch.float64)
            a.proj.weight.copy_(torch.eye(8,dtype=torch.float64));a.proj.bias.zero_()
        return a
    def test_gradient_topology(self):
        a=self.controlled(); x=[t.clone().requires_grad_() for t in self.x]; y=self.run_exchange(x,a)
        g=torch.autograd.grad(y[0][0,0,0],x[1],retain_graph=True)[0]
        self.assertTrue((g[0,::3,0]>0).all()); self.assertEqual(g[0,1::3].abs().max(),0);self.assertEqual(g[0,2::3].abs().max(),0)
        g=torch.autograd.grad(y[0][0,1,0],x[1],allow_unused=True)[0]
        self.assertTrue(g is None or g.abs().max()==0)
    def test_single_window_and_order(self):
        y=self.run_exchange(self.x[:1])[0]; torch.testing.assert_close(y,self.attn(self.x[0]),atol=1e-9,rtol=1e-9)
        before=[x.clone() for x in self.x]
        a=self.run_exchange(self.x); b=self.run_exchange(self.x[::-1])[::-1]
        for l,r in zip(a,b):torch.testing.assert_close(l,r,atol=1e-9,rtol=1e-9)
        for l,r in zip(self.x,before):self.assertTrue(torch.equal(l,r))
    def test_single_window_uses_original_block_exactly(self):
        block=Block(8,2,qk_norm=True).eval().requires_grad_(False)
        x=torch.randn(1,12,8)
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            expected=block(x)
            with patch.object(block,'forward',wraps=block.forward) as original:
                actual=global_step(block,[x],[None],3,'camera_exchange')[0]
            self.assertEqual(original.call_count,1,'single window must retain the original unsplit block path')
            self.assertTrue(torch.equal(actual,expected))

    def test_single_window_attention_avoids_query_split(self):
        a=Attention(8,2).eval().requires_grad_(False);x=torch.randn(1,12,8)
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            bank=[camera_bank(a,x,None,3)]
            with patch.object(a,'forward',wraps=a.forward) as original:
                y=exchange_attention(a,x,None,3,bank,0)
            self.assertEqual(original.call_count,1)
            self.assertTrue(torch.equal(y,a(x)))

    def test_bf16_scatter(self):
        a=Attention(8,2).eval(); x=[t.float() for t in self.x]
        with torch.autocast('cpu',dtype=torch.bfloat16): y=self.run_exchange(x,a)
        self.assertEqual(y[0].dtype,torch.bfloat16);self.assertTrue(torch.isfinite(y[0]).all())
    def test_relay_after_local_frame(self):
        b=Block(8,2).double().eval(); b.norm1=nn.Identity();b.norm2=nn.Identity();b.attn=self.controlled()
        with torch.no_grad():
            for p in b.mlp.parameters():p.zero_()
        base=[torch.zeros_like(t) for t in self.x]; changed=[t.clone() for t in base];changed[1][0,0,0]=1
        a=global_step(b,base,[None,None],3,'camera_exchange'); c=global_step(b,changed,[None,None],3,'camera_exchange')
        self.assertEqual((a[0]-c[0])[0,2].abs().max(),0)
        frame=self.controlled()
        aa=frame(a[0].reshape(2,3,8));cc=frame(c[0].reshape(2,3,8))
        self.assertGreater(float((aa-cc)[0,2].abs().max().detach()),0)
    def test_scope_validation(self):
        with self.assertRaises(ValueError): global_step(Block(8,2).eval(),self.x,[None,None],3,'wrong')
if __name__=='__main__':unittest.main()
