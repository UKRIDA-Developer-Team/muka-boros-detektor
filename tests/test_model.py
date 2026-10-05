import importlib
import importlib.util
import unittest

import torch


class ModelTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('faceage.model'), 'Model belum dibuat')
        return importlib.import_module('faceage.model')

    def test_frozen_backbone_keeps_batchnorm_statistics_during_training(self):
        m = self.module()
        model = m.create_model(weights='none', train_median=35)
        m.configure_training(model,'b1')
        m.set_training_mode(model,'b1')
        bn = next(layer for layer in model.features.modules() if isinstance(layer,torch.nn.BatchNorm2d))
        before = bn.running_mean.clone()
        output = model(torch.randn(2,3,224,224))
        output.sum().backward()
        self.assertEqual(tuple(output.shape),(2,1))
        self.assertTrue(torch.equal(before,bn.running_mean))
        self.assertTrue(all(p.grad is None for p in model.features.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.classifier.parameters()))

    def test_finetuning_only_opens_last_backbone_blocks(self):
        m = self.module()
        model = m.create_model(weights='none')
        m.configure_training(model,'m1',3)
        self.assertFalse(any(p.requires_grad for layer in model.features[:-3] for p in layer.parameters()))
        self.assertTrue(all(p.requires_grad for layer in model.features[-3:] for p in layer.parameters()))
        with self.assertRaises(ValueError): m.configure_training(model,'m1',0)

    def test_validation_transform_is_repeatable_and_normalized(self):
        from PIL import Image
        m = self.module()
        image = Image.new('RGB',(320,280),(255,0,0))
        transform = m.make_transform(training=False)
        first,second = transform(image),transform(image)
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(tuple(first.shape),(3,224,224))
        self.assertAlmostEqual(float(first[0,0,0]), (1-.485)/.229, places=5)

    def test_invalid_training_configuration_fails_before_writing_run(self):
        self.assertIsNotNone(importlib.util.find_spec('faceage.training'), 'Pelatihan belum dibuat')
        module=importlib.import_module('faceage.training')
        for override in [{'epochs':0},{'batch_size':0},{'patience':0},{'learning_rate':-1},{'loss_beta':0},
                         {'sampling':'unknown'},{'weights':'unknown'}]:
            with self.assertRaises(ValueError): module.validate_config(override)
