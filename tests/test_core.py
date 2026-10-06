import unittest
import json
from pathlib import Path
import torch
from src.models import TumorNet, objective, boundary_target
from src.train import metrics_per_slice
from src.evaluate import boundary_f1


class CoreChecks(unittest.TestCase):
    def test_exact_metrics(self):
        mask = torch.zeros(2,1,32,32)
        mask[:,:,8:18,8:18] = 1
        exact = metrics_per_slice(mask*40-20, mask)
        for values in exact.values():
            self.assertTrue(torch.allclose(values,torch.ones_like(values)))
        miss = metrics_per_slice(torch.full_like(mask,-20), mask)
        self.assertLess(float(miss['dice'].max()),1e-6)

    def test_variants_have_finite_gradients(self):
        torch.set_num_threads(2)
        x = torch.rand(2,1,64,64)
        mask = torch.zeros_like(x)
        mask[:,:,12:24,20:35] = 1
        for variant in ['baseline','softroi','srb']:
            model = TumorNet(variant)
            out = model(x)
            self.assertEqual(out['seg_logits'].shape,mask.shape)
            loss = objective(out,mask,torch.tensor([0,2]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_patient_separation(self):
        path = Path('data/processed/manifest.json')
        if not path.exists(): self.skipTest('Run src.prepare first')
        records = json.loads(path.read_text())['records']
        sets = {s:{r['pid'] for r in records if r['split']==s} for s in ['train','val','test']}
        self.assertFalse(sets['train'] & sets['val'])
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['val'] & sets['test'])
        self.assertEqual(len({r['file'] for r in records}),len(records))

    def test_boundary_interior(self):
        mask = torch.zeros(1,1,32,32);mask[:,:,8:20,8:20]=1
        edge = boundary_target(mask)
        self.assertEqual(float(edge[0,0,14,14]),0)
        self.assertEqual(float(edge[0,0,8,14]),1)
        self.assertEqual(float(edge[0,0,7,14]),1)

    def test_boundary_score(self):
        import numpy as np
        target=np.zeros((32,32),dtype=bool);target[8:18,8:18]=True
        self.assertEqual(boundary_f1(target,target),1.0)
        self.assertEqual(boundary_f1(np.zeros_like(target),target),0.0)


if __name__ == '__main__': unittest.main()
