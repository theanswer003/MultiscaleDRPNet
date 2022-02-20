import os
import copy
import time
import psutil
import pickle
import json
import numpy as np
import torch
import dnnlib
from training.dataset import  ImageFolderDataset
from training.networks import Generator, Discriminator
from training.augment import AugmentPipe
from training.loss import StyleGAN2Loss
from training.utils import setup_snapshot_image_grid, save_image_grid
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from metrics import metric_main

##################################################################################
run_dir = 'training-runs/SSK'
num_gpus = 1
batch_size = 16
batch_gpu = batch_size // num_gpus

augment_p = 0
total_kimg = 2000
ema_kimg = 5.0
ema_rampup = 0.05
ada_interval = 4
ada_target = 0.6
image_snapshot_ticks = 1
network_snapshot_ticks = 1
ada_kimg = 500
kimg_per_tick = 4
metrics= ['fid50k_full']
##################################################################################

def training_loop():
    # Initialize.
    random_seed = 0
    start_time = time.time()
    device = torch.device('cuda')
    np.random.seed(random_seed * num_gpus)
    torch.manual_seed(random_seed * num_gpus)
    torch.backends.cudnn.benchmark = True    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = False        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    training_set_kwargs = {'path': './data/SSK.zip',
                           'use_labels': False,
                           'max_size': 372,
                           'xflip': False,
                           'resolution': 1024}
    # Load training set.
    print('Loading training set...')
    training_set = ImageFolderDataset(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=0, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, sampler=training_set_sampler))
    print()
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    print('Label shape:', training_set.label_shape)
    print()

    # Construct networks.
    print('Constructing networks...')

    mapping_kwargs = {
        'num_layers': 2
    }
    synthesis_kwargs = {
        'channel_base': 32768//4,
        'channel_max': 512//4,
        'num_fp16_res': 4,
        'conv_clamp': 256
    }
    G = Generator(z_dim=512,
                  c_dim=training_set.label_dim,
                  w_dim=512,
                  img_resolution=training_set.resolution,
                  img_channels=training_set.num_channels,
                  mapping_kwargs=mapping_kwargs,
                  synthesis_kwargs=synthesis_kwargs
                  ).train().requires_grad_(False).to(device)
    D = Discriminator(
        c_dim=training_set.label_dim,
        img_resolution=training_set.resolution,
        img_channels=training_set.num_channels,
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        channel_base=32768//4,  # Overall multiplier for the number of channels.
        channel_max=512//4,  # Maximum number of channels in any layer.
        num_fp16_res=4,  # Use FP16 for the N highest resolutions.
        conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        block_kwargs={},  # Arguments for DiscriminatorBlock.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        epilogue_kwargs={'mbstd_group_size': 4},  # Arguments for DiscriminatorEpilogue.
    ).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()

    # Print network summary tables.
    z = torch.empty([batch_gpu, G.z_dim], device=device)
    c = torch.empty([batch_gpu, G.c_dim], device=device)
    img = misc.print_module_summary(G, [z, c])
    misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    print('Setting up augmentation...')
    augment_pipe = AugmentPipe(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=0,
                               contrast=0, lumaflip=0, hue=0, saturation=0).train().requires_grad_(False).to(device)
    augment_pipe.p.copy_(torch.as_tensor(augment_p))
    ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    print('Setting up training phases...')
    loss = StyleGAN2Loss(device, r1_gamma=13.1072, **ddp_modules)
    phases = []
    G_opt_kwargs = {'class_name': 'torch.optim.Adam', 'lr': 0.002, 'betas': [0, 0.99], 'eps': 1e-08}
    D_opt_kwargs = {'class_name': 'torch.optim.Adam', 'lr': 0.002, 'betas': [0, 0.99], 'eps': 1e-08}
    G_reg_interval = 4
    D_reg_interval = 16
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        phase.start_event = torch.cuda.Event(enable_timing=True)
        phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    print('Exporting sample images...')
    grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
    save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
    grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
    grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
    images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
    save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
    try:
        import torch.utils.tensorboard as tensorboard
        stats_tfevents = tensorboard.SummaryWriter(run_dir)
    except ImportError as err:
        print('Skipping tfevents export:', err)

    # Train.
    print(f'Training for {total_kimg} kimg...')
    print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        print(' '.join(fields))

        # Save image snapshot.
        if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        training_set_kwargs2 = {'class_name': 'training.dataset.ImageFolderDataset',
                                'path': './data/SSK.zip',
                                'use_labels': False,
                                'max_size': 372,
                                'xflip': False,
                                'resolution': 1024}
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs2))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs2, num_gpus=num_gpus, rank=0, device=device)
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    print()
    print('Exiting...')


if __name__ == "__main__":
    dnnlib.util.Logger(should_flush=True)
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training_loop()
