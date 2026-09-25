import tempfile,unittest,json
from unittest.mock import patch
import types
from pathlib import Path
import torch
from experiments.ours_v5.runtime import fresh_directory,prepare_inputs
class RuntimeTest(unittest.TestCase):
    def test_direct_launcher_has_no_gate_options(self):
        import subprocess,sys
        from experiments.ours_v5.runtime import ROOT
        help_text=subprocess.check_output(
            [sys.executable,'-B','-m','experiments.ours_v5','--help'],
            cwd=ROOT,text=True)
        self.assertIn('diagnose-direct',help_text)
        for removed in ('--gate','--kind','--precision','GATE_COMPLETE'):
            self.assertNotIn(removed,help_text)

    def test_direct_launcher_runs_two_modes_without_gpu_work(self):
        import importlib,os,sys
        from unittest.mock import MagicMock
        from experiments.ours_v5.runtime import ROOT
        entry=importlib.import_module('experiments.ours_v5.__main__')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result';calls=[]
            def prepare(_scene,_frame_list,path,frames):
                self.assertEqual(frames,100)
                path.write_bytes(b'cpu-only input fixture')
                return {'frame_ids':[f'{i:06d}' for i in range(100)],'preprocessing':{'elapsed_seconds':0.0}}
            def run(command,**_kwargs):
                mode=command[command.index('--mode')+1]
                self.assertEqual(command[command.index('--window-size')+1],'60')
                self.assertEqual(command[command.index('--overlap')+1],'30')
                self.assertEqual(command[command.index('--batch-size')+1],'2')
                self.assertEqual(command[command.index('--frames')+1],'100')
                self.assertNotIn('--capture',command)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'999')
                target=Path(command[command.index('--output')+1]);target.mkdir()
                (target/'COMPLETE.json').write_text(json.dumps({'status':'complete'}))
                (target/'run_manifest.json').write_text(json.dumps({'timing':{},'peak_allocated_bytes':0,
                    'peak_reserved_bytes':0,'cpu_peak_rss_bytes':0,'gpu_uuid':'CPU-ONLY'}))
                (target/'trajectory_metrics.json').write_text(json.dumps({'ate_rmse_m':0.0}))
                calls.append(mode)
            sampler=MagicMock()
            with patch.dict(os.environ,{},clear=False),patch.object(sys,'argv',['ours_v5','diagnose-direct','--gpu','999','--output',str(output)]),\
                 patch.object(entry,'source_identity',return_value={'commit':'fixed-test-commit','status':'','sha256':{}}),\
                 patch.object(entry,'preflight',return_value={'gpu_uuid':'CPU-ONLY'}),\
                 patch.object(entry,'data_identity',return_value={'fixed_frames':100}),\
                 patch.object(entry,'prepare_inputs',side_effect=prepare),\
                 patch.object(entry.subprocess,'Popen',return_value=sampler),\
                 patch.object(entry.subprocess,'run',side_effect=run):
                entry.main()
            self.assertEqual(calls,['independent','camera_exchange'])
            self.assertTrue((output/'COMPLETE.json').is_file())
            self.assertTrue((output/'summary.json').is_file())
            self.assertFalse((output/'GATE_COMPLETE.json').exists())
            self.assertEqual(list(output.rglob('window_*.pt')),[])

    def test_all_new_modules_compile(self):
        from experiments.ours_v5.runtime import ROOT
        for folder in ('experiments/ours_v5','vggt/v5'):
            for path in (ROOT/folder).glob('*.py'):
                compile(path.read_text(),str(path),'exec')

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';fresh_directory(p);(p/'keep').write_text('old')
            with self.assertRaises(FileExistsError):fresh_directory(p)
            self.assertEqual((p/'keep').read_text(),'old')
    def test_preflight_rejects_busy_gpu_and_low_disk(self):
        from experiments.ours_v5.runtime import preflight
        def command(args,**kwargs):
            if '--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu' in args:return 'GPU-test, NVIDIA H20, 0, 97000, 0'
            if '--query-compute-apps=gpu_uuid,pid,process_name' in args:return 'GPU-test, 123, python'
            return ''
        with tempfile.TemporaryDirectory() as d,patch('socket.gethostname',return_value='VM-0-11-ubuntu'),patch('getpass.getuser',return_value='ubuntu'),patch('subprocess.check_output',side_effect=command):
            with self.assertRaisesRegex(RuntimeError,'occupied'):preflight('0',d)
        def idle(args,**kwargs):
            if '--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu' in args:return 'GPU-test, NVIDIA H20, 0, 97000, 0'
            return ''
        with tempfile.TemporaryDirectory() as d,patch('socket.gethostname',return_value='VM-0-11-ubuntu'),patch('getpass.getuser',return_value='ubuntu'),patch('subprocess.check_output',side_effect=idle),patch('shutil.disk_usage',return_value=types.SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError,'disk'):preflight('0',d)

    def test_prepare_missing_fixed_list_stops(self):
        with tempfile.TemporaryDirectory() as d:
            ids=Path(d)/'ids.json';ids.write_text(json.dumps({'frame_ids':['a','a']}))
            with self.assertRaises(ValueError):prepare_inputs(Path(d),ids,Path(d)/'input.pt',2)
if __name__=='__main__':unittest.main()
