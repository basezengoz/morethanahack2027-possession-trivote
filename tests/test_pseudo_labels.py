import importlib.util
import unittest
import tempfile
from pathlib import Path
import numpy as np
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from possession.train import augment, fit, load_checkpoint, masked_loss, validate_split, windows
    from possession.model import N_RAW, TAG_START, SCHEMA
    torch.set_num_threads(1)

@unittest.skipUnless(HAS_TORCH,'optional training dependency is not installed')
class TrainingTest(unittest.TestCase):
    def test_final_fit_requires_explicit_fixed_epoch_mode(self):
        m={'id':'a','raw':np.ones((20,N_RAW),np.float32),'labels':np.array([0,1]*10),
           'mask':np.ones(20,bool),'blocks':[(0,20)]}
        cfg=dict(seed=1,channels=2,epochs=1,learning_rate=.001,center_frames=20,
                 context_frames=256,max_train_windows=1,team_swap=0.,tag_dropout=0.,
                 missing_block_probability=0.,missing_block_frames=2)
        with self.assertRaises(ValueError): fit([m],[],cfg)
        cfg['fixed_epoch_final_training']=True
        with self.assertRaises(ValueError): fit([m,m],[],cfg)
        _,_,_,stats=fit([m],[],cfg)
        self.assertEqual(stats['epochs'],1)
        self.assertEqual(stats['epoch_validation_proxy'],[])

    def test_masked_loss_ignores_unknown_and_dead_head_if_untrained(self):
        dead=torch.tensor([20.,-20.,4.],requires_grad=True)
        team=torch.tensor([[3.,0.],[0.,3.],[0.,0.]],requires_grad=True)
        y=torch.tensor([0,1,-1]); mask=torch.tensor([True,True,False])
        loss=masked_loss(dead,team,y,mask,False); loss.backward()
        self.assertIsNone(dead.grad); self.assertTrue(torch.equal(team.grad[2],torch.zeros(2)))
        self.assertIsNone(masked_loss(dead,team,y,torch.zeros(3,dtype=torch.bool),False))

    def test_match_split_and_context(self):
        m={'id':'a','raw':np.ones((20,N_RAW),np.float32),'labels':np.array([0,1]*10),'mask':np.ones(20,bool),'blocks':[(0,8),(8,20)]}
        with self.assertRaises(ValueError): validate_split([m],[m])
        for lo,hi,left,right in windows(m,4,256):
            self.assertTrue((0<=lo<=left<right<=hi<=8) or (8<=lo<=left<right<=hi<=20))
        with self.assertRaises(ValueError): list(windows(m,4,10))

    def test_fixed_synthetic_fit_and_augment(self):
        m={'id':'a','raw':np.ones((20,N_RAW),np.float32),'labels':np.array([0,1]*10),'mask':np.ones(20,bool),'blocks':[(0,20)]}
        val={**m,'id':'b','raw':np.full((20,N_RAW),999,dtype=np.float32)}
        cfg=dict(seed=1,channels=2,epochs=1,learning_rate=.001,center_frames=20,context_frames=256,max_train_windows=1,team_swap=0.,tag_dropout=1.,missing_block_probability=0.,missing_block_frames=2)
        augmented,_=augment(m['raw'],m['labels'],np.random.default_rng(1),cfg)
        self.assertTrue(np.all(augmented[:,TAG_START:]==0))
        _,norm,_,stats=fit([m],[val],cfg)
        self.assertEqual(stats['steps'],1); self.assertFalse(stats['dead_head_trained'])
        self.assertEqual(stats['unique_supervised_samples_seen'],20)
        self.assertEqual(stats['supervision_coverage'],1.)
        self.assertEqual(stats['candidate_windows'],1)
        self.assertEqual(stats['epoch_validation_proxy'][0]['support'],[10,10])
        self.assertIsNone(stats['epoch_validation_proxy'][0]['spell_f1'])
        self.assertFalse(stats['final_selection_eligible'])
        np.testing.assert_allclose(norm.mean,1)

    def test_missing_augmentation_uses_inference_values_and_masks(self):
        raw=np.ones((20,N_RAW),np.float32)
        cfg=dict(team_swap=0.,tag_dropout=0.,missing_block_probability=1.,missing_block_frames=4)
        out,_=augment(raw,np.zeros(20,np.int64),np.random.default_rng(5),cfg)
        offset=0; columns={}
        for name,width in SCHEMA:
            columns[name]=slice(offset,offset+width); offset+=width
        masked=out[:,columns['ball_available']][:,0]==0
        self.assertTrue(masked.any())
        for name in ('control_support','nearby_player_count','ball_observed','ball_interpolated','ball_rejected'):
            self.assertTrue((out[masked,columns[name]]==0).all())
        for name in ('nearest_distance','ball_speed','boundary_distance'):
            self.assertTrue(np.isnan(out[masked,columns[name]]).all())

    def test_normalization_excludes_unscored_context_and_d_supervision_gradients(self):
        raw=np.ones((20,N_RAW),np.float32); raw[:5]=999; raw[15:]=999
        mask=np.zeros(20,bool); mask[5:15]=True
        m={'id':'a','raw':raw,'labels':np.zeros(20,np.int64),'mask':mask,'blocks':[(5,15)]}
        cfg=dict(seed=1,channels=2,epochs=1,learning_rate=.001,center_frames=20,context_frames=256,max_train_windows=1,team_swap=0.,tag_dropout=0.,missing_block_probability=0.,missing_block_frames=2)
        _,norm,_,_=fit([m],[{**m,'id':'b'}],cfg)
        np.testing.assert_allclose(norm.mean,1)
        m['mask'][0]=True
        with self.assertRaises(ValueError): fit([m],[{**m,'id':'b'}],cfg)
        dead=torch.zeros(3,requires_grad=True); team=torch.zeros((3,2),requires_grad=True)
        loss=masked_loss(dead,team,torch.tensor([0,2,-1]),torch.ones(3,dtype=torch.bool),True,
                         torch.tensor([True,True,False]))
        loss.backward()
        self.assertGreater(dead.grad[0],0); self.assertLess(dead.grad[1],0)
        self.assertEqual(dead.grad[2],0)
        self.assertTrue(torch.equal(team.grad[1:],torch.zeros((2,2))))
        dead=torch.zeros(2,requires_grad=True); team=torch.zeros((2,2),requires_grad=True)
        loss=masked_loss(dead,team,torch.tensor([0,1]),torch.ones(2,dtype=torch.bool),True,
                         torch.zeros(2,dtype=torch.bool))
        loss.backward()
        self.assertIsNone(dead.grad)

    def test_dead_head_requires_reliable_positive_and_negative_masks(self):
        cfg=dict(seed=1,channels=2,epochs=1,learning_rate=.001,center_frames=20,
                 context_frames=256,max_train_windows=1,team_swap=0.,tag_dropout=0.,
                 missing_block_probability=0.,missing_block_frames=2)
        labels=np.array([0,2]*10)
        m={'id':'a','raw':np.ones((20,N_RAW),np.float32),'labels':labels,
           'mask':np.ones(20,bool),'blocks':[(0,20)],
           'dead_mask':np.array([True,False]*10)}
        _,_,_,stats=fit([m],[{**m,'id':'b'}],cfg)
        self.assertFalse(stats['dead_head_trained'])
        self.assertEqual(stats['dead_supervision_counts'],{'negative_live':10,'positive_D':0})
        m['dead_mask']=np.ones(20,bool)
        _,_,_,stats=fit([m],[{**m,'id':'b'}],cfg)
        self.assertTrue(stats['dead_head_trained'])
        without_dead_mask={key:value for key,value in m.items() if key != 'dead_mask'}
        _,_,_,stats=fit([without_dead_mask],[{**without_dead_mask,'id':'b'}],cfg)
        self.assertFalse(stats['dead_head_trained'])
        m['dead_mask']=np.ones(19,bool)
        with self.assertRaises(ValueError): fit([m],[{**m,'id':'b'}],cfg)

    def test_trusted_checkpoint_roundtrip_and_schema_guard(self):
        m={'id':'a','raw':np.ones((20,N_RAW),np.float32),'labels':np.array([0,1]*10),'mask':np.ones(20,bool),'blocks':[(0,20)]}
        cfg=dict(seed=1,channels=2,epochs=1,learning_rate=.001,center_frames=20,context_frames=256,max_train_windows=1,team_swap=0.,tag_dropout=0.,missing_block_probability=0.,missing_block_frames=2)
        model,norm,opt,stats=fit([m],[{**m,'id':'b'}],cfg)
        checkpoint={'schema':SCHEMA,'config':cfg,'model':model.state_dict(),'optimizer':opt.state_dict(),
                    'normalizer':{'mean':norm.mean,'scale':norm.scale},'stats':stats}
        directory=Path('data/runs/training'); directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            path=Path(tmp)/'synthetic.pt'; torch.save(checkpoint,path)
            restored,normalizer,metadata=load_checkpoint(path)
            np.testing.assert_array_equal(normalizer.mean,norm.mean)
            self.assertFalse(metadata['stats']['dead_head_trained'])
            for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,restored.state_dict()[k]))
            checkpoint['schema']=(('invalid',1),); torch.save(checkpoint,path)
            with self.assertRaises(ValueError): load_checkpoint(path)

if __name__=='__main__': unittest.main()
