import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['SAREdgeNet','UNet','AttentionUNet','DeepLabV3Plus','PSPNet','FPN']


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_epochs(iterable,output):
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    durations = []
    output.write_text('[]',encoding='utf-8')
    for value in iterable:
        synchronize()
        start = time.perf_counter()
        try:
            yield value
        finally:
            synchronize()
            durations.append(time.perf_counter()-start)
            output.write_text(json.dumps(durations,indent=2),encoding='utf-8')


def run_script(path,arguments):
    previous = sys.argv[:]
    previous_path = sys.path[:]
    sys.path.insert(0,str(Path(path).resolve().parent))
    previous_path = sys.path[:]
    sys.path.insert(0,str(Path(path).resolve().parent))
    sys.argv = [str(path),*map(str,arguments)]
    try:
        result = runpy.run_path(str(path),run_name='__main__')
        del result
    finally:
        sys.argv = previous
        sys.path[:] = previous_path
        sys.path[:] = previous_path
    gc.collect()


def model_metadata(method,config):
    if method=='SAREdgeNet':
        from .models import SAREdgeNetContourClassifier
        model = SAREdgeNetContourClassifier(pretrained=False)
        checkpoint = config.get('checkpoint',str(Path(config.get('classifier_dir',str(Path(config['output_dir'])/'step5')))/'best_classifier.pt'))
        key = 'model_state'
    else:
        sys.path.insert(0,str(ROOT/'对比实验代码'))
        module = importlib.import_module(method+'_Step1')
        factory = getattr(module,method)
        options = {'in_channels':2 if method in ['DeepLabV3Plus','PSPNet','FPN'] else 1,'n_classes':2}
        if method=='DeepLabV3Plus':
            options['pretrained'] = False
        model = factory(**options)
        checkpoint = config['model_path'] if method=='DeepLabV3Plus' else config['model_save_path']
        key = 'model' if method=='DeepLabV3Plus' else 'model_state_dict'
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if method in ['PSPNet','FPN']:
        from .sar_dem_baselines import validate_checkpoint
        validate_checkpoint(saved,model)
    elif method in ['UNet','AttentionUNet','DeepLabV3Plus']:
        from .comparison_models_v2 import validate_model_checkpoint
        validate_model_checkpoint(saved,model)
    state = saved.get(key,saved)
    model.load_state_dict(state)
    result = {'model_parameters':sum(p.numel() for p in model.parameters()),
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'checkpoint':str(Path(checkpoint).resolve()),
              'checkpoint_bytes':Path(checkpoint).stat().st_size}
    if method in ['PSPNet','FPN','UNet','AttentionUNet','DeepLabV3Plus']:
        result['model_architecture'] = saved['model_metadata']['architecture_id']
        result['model_input_spec'] = saved['model_metadata']['input_spec']
    digest = hashlib.sha256()
    with Path(checkpoint).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    result['checkpoint_sha256'] = digest.hexdigest()
    del model,saved,state
    gc.collect()
    return result


def benchmark_one(spec,work):
    method = spec['method']
    if method not in METHODS:
        raise ValueError('Unsupported deep learning method')
    inference = spec['inference_config']
    config = json.loads(Path(inference).read_text(encoding='utf-8')) if isinstance(inference,str) else inference.copy()
    work = Path(work).resolve()
    work.mkdir(parents=True,exist_ok=True)
    scope = spec.get('scope','end_to_end')
    if scope not in ['end_to_end','prepared_input_inference']:
        raise ValueError('Unsupported timing scope')
    commands = []
    metadata_config = config.copy()
    if method=='SAREdgeNet':
        from Run_Step1 import commands as pipeline_commands
        config['checkpoint'] = metadata_config.get('checkpoint',str(Path(metadata_config.get('classifier_dir',str(Path(metadata_config['output_dir'])/'step5')))/'best_classifier.pt'))
        config['output_dir'] = str(work/'pipeline')
        if scope=='end_to_end':
            config.pop('processed_sar',None)
            config.pop('contour_dir',None)
            config.pop('support_mask',None)
            steps = [1,2,3,4,6,7,8,9]
        else:
            if not config.get('processed_sar') or not config.get('contour_dir'):
                raise ValueError('Prepared-input SAREdgeNet timing needs processed_sar and contour_dir')
            steps = [6,7,8,9]
        commands = [(Path(command[1]),command[2:]) for _,command in pipeline_commands(config,steps)]
        metadata_config['checkpoint'] = config['checkpoint']
    else:
        if scope=='end_to_end':
            if not spec.get('preprocess_input'):
                raise ValueError('End-to-end network timing needs preprocess_input (calibrated, terrain-corrected SAR)')
            commands.append((ROOT/'主实验代码/SAREdgeNet_Step1.py',['--input',spec['preprocess_input'],'--output',work/'normalized.tif']))
            geometry_args = ['--input',work/'normalized.tif','--output',work/'sar.tif']
            if spec.get('target_crs'):
                geometry_args += ['--target-crs',spec['target_crs'],'--resolution',20]
            commands.append((ROOT/'主实验代码/SAREdgeNet_Step2.py',geometry_args))
            config['tif_path'] = str(work/'sar.tif')
            if method in ['DeepLabV3Plus','PSPNet','FPN']:
                dem = spec.get('dem_source') or config.get('dem_preprocessed') or config.get('dem_path')
                if not dem:
                    raise ValueError('SAR+DEM network timing needs a DEM')
                commands.append((ROOT/'主实验代码/SAREdgeNet_Step3.py',['--dem',dem,'--reference',work/'sar.tif','--output',work/'dem.tif']))
                config['dem_preprocessed' if method=='DeepLabV3Plus' else 'dem_path'] = str(work/'dem.tif')
        config['output_path'] = str(work/'prediction.tif')
        config_path = work/'effective_inference_config.json'
        config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        commands.append((ROOT/f'对比实验代码/{method}_Step2.py',['--config',config_path]))
    torch.set_num_threads(int(spec.get('cpu_threads',4)))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(0)
        synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for script,arguments in commands:
        run_script(script,arguments)
    synchronize()
    elapsed = time.perf_counter()-start
    allocated = torch.cuda.max_memory_allocated() if cuda else None
    reserved = torch.cuda.max_memory_reserved() if cuda else None
    result = {'method':method,'scope':scope,'seconds':elapsed,'model_warmup':False,
              'device':torch.cuda.get_device_name(0) if cuda else 'cpu',
              'gpu_peak_allocated_bytes':allocated,'gpu_peak_reserved_bytes':reserved,
              'gpu_memory_basis':'PyTorch allocator peak; excludes CUDA context and non-PyTorch allocations' if cuda else 'not_applicable_cpu',
              'cpu_threads':torch.get_num_threads(),'torch_version':torch.__version__,
              'cuda_runtime':torch.version.cuda,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'commands':[{'script':str(path),'arguments':list(map(str,args))} for path,args in commands]}
    import rasterio
    timed_input = config['tif_path'] if method!='SAREdgeNet' else (config['processed_sar'] if scope=='prepared_input_inference' else work/'pipeline/step2/sar.tif')
    with rasterio.open(timed_input) as source:
        result['input_grid'] = {'width':source.width,'height':source.height,'crs':str(source.crs),
                                'transform':list(source.transform)[:6]}
        result['input_pixels'] = source.width*source.height
    result.update(model_metadata(method,metadata_config))
    return result


def summary(raw,epoch_times=None,planned_epochs=None):
    values = raw['seconds']
    if not values or any(not math.isfinite(value) or value<=0 for value in values):
        raise ValueError('Timing samples must be positive')
    low,high = min(values),max(values)
    result = {**raw,'minimum_seconds':low,'maximum_seconds':high,
              'midpoint_seconds':(low+high)/2,'half_range_seconds':(high-low)/2}
    runs = raw.get('runs',[])
    if runs:
        parameters = {r['model_parameters'] for r in runs}
        scopes = {r['scope'] for r in runs}
        devices = {r['device'] for r in runs}
        grids = {json.dumps(r['input_grid'],sort_keys=True) for r in runs}
        checkpoints = {r.get('checkpoint_sha256') for r in runs}
        if len(parameters)!=1 or len(scopes)!=1 or len(devices)!=1 or len(grids)!=1 or len(checkpoints)!=1:
            raise ValueError('Runs differ in architecture, checkpoint, timing scope, device or grid')
        result.update(model_parameters=runs[0]['model_parameters'],parameters_millions=runs[0]['model_parameters']/1e6,
                      scope=runs[0]['scope'],device=runs[0]['device'],checkpoint=runs[0]['checkpoint'],
                      input_grid=runs[0]['input_grid'],input_pixels=runs[0]['input_pixels'])
        for key in ['gpu_peak_allocated_bytes','gpu_peak_reserved_bytes']:
            known = [r[key] for r in runs if r[key] is not None]
            result[key.replace('_bytes','_gib')] = max(known)/1024**3 if known else None
    if epoch_times is not None:
        if not planned_epochs or planned_epochs<1 or not epoch_times or any(not math.isfinite(t) or t<=0 for t in epoch_times):
            raise ValueError('Provide positive sampled epoch durations and planned_epochs')
        lower,upper = min(epoch_times)*planned_epochs,max(epoch_times)*planned_epochs
        result['training_time_estimate'] = {'minimum_seconds':lower,'maximum_seconds':upper,
                                             'midpoint_seconds':(lower+upper)/2,'half_range_seconds':(upper-lower)/2,
                                             'sampled_epochs':len(epoch_times),'planned_epochs':planned_epochs,
                                             'kind':'extrapolation; epoch time includes training, validation and checkpoint I/O'}
    return result
