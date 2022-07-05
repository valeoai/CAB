import os
import sys
import json
import time
import random
import pathlib
import warnings

import dill
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("PS")

from tqdm import tqdm
from torch import nn, optim, utils
from tensorboardX import SummaryWriter

import evaluation
import visualization

from options import deep_asdict, Options, OptionsDict

from model.dataset import collate, EnvironmentDataset
from model.trajectron import Trajectron
from model.model_utils import cyclical_lr
from model.model_registrar import ModelRegistrar

nu_path = '/root/workspace/nuscenes-devkit/python-sdk/'
sys.path.insert(0,nu_path)

from nuscenes.eval.prediction.metrics import MinADEK, MinFDEK, RowMean
from nuscenes.eval.prediction.data_classes import Prediction

# torch.autograd.set_detect_anomaly(True)

def deep_asdict(opt):
    out_opt = {}
    for k in opt:
        v = opt[k]
        if isinstance(v, OptionsDict):
            out_opt[k] = deep_asdict(dict(v))
        elif isinstance(v, dict):
            out_opt[k] = deep_asdict(v)
        else:
            out_opt[k] = v
    return out_opt

hyperparams = deep_asdict(Options().options)
device = hyperparams['device']

if not torch.cuda.is_available() or device == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        device = 'cuda:0'

    device = torch.device(device)

if hyperparams['eval_device'] is None:
    eval_device = torch.device('cpu')
else:
    eval_device = torch.device(hyperparams['eval_device'])

# This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
torch.cuda.set_device(device)

seed = hyperparams['seed']
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fetch_batch(generator):
    try:
        batch = next(generator)
        not_empty = True
    except StopIteration:
        batch = None
        not_empty = False
    return batch, not_empty


def part2train(model_registrar, part='discriminator'):
    if part == 'discriminator':
        for para in model_registrar.get_all_but_name_match(['discri']).parameters():
            para.requires_grad = False
        for para in model_registrar.get_name_match(['discri']).parameters():
            para.requires_grad = True
    elif part == 'generator':
        for para in model_registrar.get_name_match(['discri']).parameters():
            para.requires_grad = False
        for para in model_registrar.get_all_but_name_match(['discri']).parameters():
            para.requires_grad = True
    else:
        raise KeyError



def main():

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % hyperparams['batch_size'])
    print('| device: %s' % device)
    print('| eval_device: %s' % eval_device)
    print('| Offline Scene Graph Calculation: %s' % hyperparams['offline_scene_graph'])
    print('| EE state_combine_method: %s' % hyperparams['edge_state_combine_method'])
    print('| EIE scheme: %s' % hyperparams['edge_influence_combine_method'])
    print('| dynamic_edges: %s' % hyperparams['dynamic_edges'])
    print('| robot node: %s' % hyperparams['incl_robot_node'])
    print('| edge_addition_filter: %s' % hyperparams['edge_addition_filter'])
    print('| edge_removal_filter: %s' % hyperparams['edge_removal_filter'])
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    log_writer = None
    model_dir = None
    if not hyperparams['debug']:
        # Create the log and model directiory if they're not present.
        if hyperparams['resume'] is None:
            ts_tag = 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + hyperparams['log_tag']
            model_dir = os.path.join(hyperparams['log_dir'],
                                     ts_tag)
            Options().options['ts_tag'] = ts_tag

        else:

            ## Starting from pretrained model
            #if hyperparams["halentent"]["load_model"] != "None":

            parent_dir = os.path.join(hyperparams['log_dir'],
                                     hyperparams['ts_tag'])
            model_dir = os.path.join(parent_dir, 'resume,%d' % int(hyperparams['resume']))


        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        # Save config to model directory
        Options().save(os.path.join(model_dir, 'config.yaml'))
        # with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
        #     json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    # Load training and evaluation environments and scenes
    train_scenes = []
    train_data_path = os.path.join(hyperparams['data_dir'], hyperparams['train_data_dict'])
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in hyperparams['override_attention_radius']:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if hyperparams['scene_freq_mult_train'] else None

    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=not hyperparams['incl_robot_node'])

    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue

        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if device is 'cpu' else True,
                                                     batch_size=hyperparams['batch_size'],
                                                     shuffle=True,
                                                     num_workers=hyperparams['preprocess_workers'])
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader


    print(f"Loaded training data from {train_data_path}")

    eval_scenes = []
    eval_scenes_sample_probs = None
    if hyperparams['eval_every'] is not None:
        eval_data_path = os.path.join(hyperparams['data_dir'], hyperparams['eval_data_dict'])
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if hyperparams['scene_freq_mult_eval'] else None

        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not hyperparams['incl_robot_node'])
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue

            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False if eval_device is 'cpu' else True,
                                                         batch_size=hyperparams['eval_batch_size'],
                                                         shuffle=True,
                                                         num_workers=hyperparams['preprocess_workers'])
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Loaded evaluation data from {eval_data_path}")

    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Evaluation Scene {i}")


    model_registrar = ModelRegistrar(model_dir, device)

    #if hyperparams['resume'] is not None:
    #    print("Loading ckpt for epoch %s" % hyperparams['resume'])
    #    model_registrar.load_models(int(hyperparams['resume']), parent_dir)


    if hyperparams["halentnet"]["load_model"] == "no_model":
        model_registrar = ModelRegistrar(model_dir, device)

        if hyperparams['resume'] is not None:
            print("Loading ckpt for epoch %s" % hyperparams['resume'])
            model_registrar.load_models(int(hyperparams['resume']), parent_dir)
    else:
        model_registrar = ModelRegistrar(hyperparams["halentnet"]["load_model"], device)

        if hyperparams["halentnet"]['checkpoint'] > 0:
            print("Loading ckpt for epoch %s" % hyperparams["halentnet"]['checkpoint'])
            model_registrar.load_models(int(hyperparams["halentnet"]['checkpoint']))



    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            device)

    creative_mode = '_'.join([hyperparams["halentnet"]["creative_t"], hyperparams["halentnet"]["creative_loss"]])

    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()
    print('Created Training Model.')

    eval_trajectron = None
    if hyperparams['eval_every'] is not None or hyperparams['vis_every'] is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     eval_device)
        eval_trajectron.set_environment(eval_env)
        eval_trajectron.set_annealing_params()
    print('Created Evaluation Model.')

    # lr_scheduler_d and optimizer_d added for HalentNet discriminator
    optimizer = dict()
    optimizer_d = dict()
    lr_scheduler = dict()
    lr_scheduler_d = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}], lr=hyperparams['learning_rate'])
        optimizer_d[node_type] = optim.Adam(model_registrar.get_name_match('discrim').parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
            lr_scheduler_d[node_type] = optim.lr_scheduler.ExponentialLR(optimizer_d[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])
            lr_scheduler_d[node_type] = optim.lr_scheduler.ExponentialLR(optimizer_d[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])


    #################################
    #        RESUMING TRAINING      #
    #################################

    # Resume model
    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    if hyperparams['resume'] is not None:
        curr_iter_dict = json.load(open(os.path.join(parent_dir,
                                                     "curr_iter_%d.json" % int(hyperparams['resume']))))
        curr_iter_node_type.update(curr_iter_dict)
        annealers_state_dict = torch.load(os.path.join(parent_dir, 'annealers_%d.pt' % int(hyperparams['resume'])),
                                          map_location=device)

        lr_scheduler_state_dict = torch.load(os.path.join(parent_dir, 'lr_scheduler_%d.pt' % int(hyperparams['resume'])),
                                             map_location=device)
        lr_scheduler_d_state_dict = torch.load(os.path.join(parent_dir, 'lr_scheduler_d_%d.pt' % int(hyperparams['resume'])),
                                             map_location=device)
        optim_state_dict = torch.load(os.path.join(parent_dir, 'optim_%d.pt' % int(hyperparams['resume'])),
                                      map_location=device)
        optim_d_state_dict = torch.load(os.path.join(parent_dir, 'optim_d_%d.pt' % int(hyperparams['resume'])),
                                      map_location=device)

        for node_type in trajectron.node_models_dict:
            node_model = trajectron.node_models_dict[node_type]
            for name, schedul in zip(node_model.annealed_vars, node_model.schedulers):
                sd = annealers_state_dict[node_type.name][name]
                schedul.load_state_dict(sd)
            lr_scheduler[node_type].load_state_dict(lr_scheduler_state_dict[node_type.name])
            lr_scheduler_d[node_type].load_state_dict(lr_scheduler_d_state_dict[node_type.name])
            optimizer[node_type].load_state_dict(optim_state_dict[node_type.name])
            for state in optimizer[node_type].state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            optimizer_d[node_type].load_state_dict(optim_d_state_dict[node_type.name])
            for state in optimizer_d[node_type].state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

    start_epoch = 1
    if hyperparams['resume'] is not None:
        start_epoch = int(hyperparams['resume'])+1

    for epoch in range(start_epoch, hyperparams['train_epochs'] + 1):
        model_registrar.to(device)
        train_dataset.augment = hyperparams['augment']

        #################################
        #         TRAINING GAN          #
        #################################
        if True:
            node_type = 'VEHICLE'
            data_loader = train_data_loader[node_type]
            curr_iter = curr_iter_node_type[node_type]

            pbar = tqdm(data_loader, ncols=80)
            pbar_gen = iter(pbar) # Added for Halentnet

            d_loss = g_loss = train_loss = torch.zeros(1)
            not_empty = True
            while not_empty:
                trajectron.set_curr_iter(curr_iter)
                trajectron.step_annealers(node_type)

                # ---------- discriminator ----------
                optimizer_d[node_type].zero_grad()
                part2train(model_registrar, 'discriminator')

                n_d_steps = 1
                for _ in range(n_d_steps):
                    batch, not_empty = fetch_batch(pbar_gen)
                    if not not_empty: break

                    d_loss, dc_loss, d_real, d_fake = trajectron.gan_d_loss(batch, node_type, grid_std=hyperparams["halentnet"]["grid_std"], grid_max=hyperparams["halentnet"]["grid_max"])
                    (d_loss + dc_loss).backward()
                    optimizer_d[node_type].step()

                # ---------- generator ----------
                optimizer[node_type].zero_grad()
                part2train(model_registrar, 'generator')

                n_g_steps = 1
                for _ in range(n_g_steps):
                    batch, not_empty = fetch_batch(pbar_gen)
                    if not not_empty: break

                    g_loss, c_loss = trajectron.gan_g_loss(batch, node_type, grid_std=hyperparams["halentnet"]["grid_std"],
                                                           grid_max=hyperparams["halentnet"]["grid_max"])
                    (1/2 * hyperparams["halentnet"]["g_factor"] * (g_loss + c_loss)).backward()
                    g_loss, creative_loss = trajectron.gan_g_loss(batch, node_type, grid_std=hyperparams["halentnet"]["grid_std"],
                                                                  grid_max=hyperparams["halentnet"]["grid_max"], creative=creative_mode)
                    (1/2 * hyperparams["halentnet"]["g_factor"] * (g_loss + hyperparams["halentnet"]["creative_lambda"] * creative_loss)).backward()
                    optimizer[node_type].step()

                # ---------- supervision ----------
                optimizer[node_type].zero_grad()
                part2train(model_registrar, 'generator')

                n_s_steps = 1
                for _ in range(n_s_steps):
                    # BUG here in HalentNet ?
                    #batch, gen_not_empty = fetch_batch(pbar_gen)
                    batch, not_empty = fetch_batch(pbar_gen)
                    if not not_empty: break
                    train_loss = trajectron.train_loss(batch, node_type)
                    train_loss.backward()
                    optimizer[node_type].step()

                f'TL: {train_loss.item():.2f}  DL: {d_loss.item():.2f} GL: {g_loss.item():.2f}'
                pbar.set_description(f"Epoch {epoch} TL: {train_loss.item():.2f}  DL: {d_loss.item():.2f} GL: {g_loss.item():.2f}")

                # Clipping gradients.
                if hyperparams['grad_clip'] is not None:
                    nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])
                optimizer[node_type].step()

                # Stepping forward the learning rate scheduler and annealers.
                lr_scheduler[node_type].step()
                lr_scheduler_d[node_type].step()

                if not hyperparams['debug']:
                    log_writer.add_scalar(f"{node_type}/train/learning_rate",
                                          lr_scheduler[node_type].get_lr()[0],
                                          curr_iter)
                    if n_d_steps > 0: log_writer.add_scalar(f"{node_type}/train/gan_d_loss", d_loss, curr_iter)
                    if n_g_steps > 0: log_writer.add_scalar(f"{node_type}/train/gan_g_loss", g_loss, curr_iter)
                    if n_s_steps > 0: log_writer.add_scalar(f"{node_type}/train/sup_loss", train_loss, curr_iter)

                curr_iter += 1
            curr_iter_node_type[node_type] = curr_iter

        train_dataset.augment = False
        if hyperparams['eval_every'] is not None or hyperparams['vis_every'] is not None:
            eval_trajectron.set_curr_iter(epoch)

        #################################
        #        VISUALIZATION          #
        #################################
        if hyperparams['vis_every'] is not None and not hyperparams['debug'] and epoch % hyperparams['vis_every'] == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict random timestep to plot for train data set
                if hyperparams['scene_freq_mult_viz']:
                    scene = np.random.choice(train_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(train_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = trajectron.predict(scene,
                                                 timestep,
                                                 ph,
                                                 min_future_timesteps=ph,
                                                 z_mode=True,
                                                 gmm_mode=True,
                                                 all_z_sep=False,
                                                 full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('train/prediction', fig, epoch)

                model_registrar.to(eval_device)
                # Predict random timestep to plot for eval data set
                if hyperparams['scene_freq_mult_viz']:
                    scene = np.random.choice(eval_scenes, p=eval_scenes_sample_probs)
                else:
                    scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      num_samples=20,
                                                      min_future_timesteps=ph,
                                                      z_mode=False,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction', fig, epoch)

                # Predict random timestep to plot for eval data set
                predictions = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      min_future_timesteps=ph,
                                                      z_mode=True,
                                                      gmm_mode=True,
                                                      all_z_sep=True,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction_all_z', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if hyperparams['eval_every'] is not None and epoch % hyperparams['eval_every'] == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    eval_loss = []
                    print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
                    pbar = tqdm(data_loader, ncols=80)
                    for batch in pbar:
                        eval_loss_node_type = eval_trajectron.eval_loss(batch, node_type)
                        pbar.set_description(f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
                        eval_loss.append({node_type: {'nll': [eval_loss_node_type]}})
                        del batch

                    evaluation.log_batch_errors(eval_loss,
                                                log_writer,
                                                f"{node_type}/eval_loss",
                                                epoch)


                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                if hyperparams['eval_type'] == "nuscenes":
                    eval_batch_errors = []
                    k_to_report = [1,5,10]
                    ns_metrics = [
                                    {'name': 'minFDE', "cls":MinFDEK(k_to_report, [RowMean()])},
                                    {'name': 'minADE', "cls":MinADEK(k_to_report, [RowMean()])},

                    ]
                    pbar = tqdm(eval_scenes, desc='MM Evaluation', ncols=80)
                    for scene in pbar:
                        timesteps = scene.sample_timesteps(scene.timesteps)
                        mode_probs, predictions = eval_trajectron.predict(scene,
                                                                             timesteps,
                                                                             ph,
                                                                             all_z_sep=True,
                                                                             gmm_mode=True,
                                                                             min_future_timesteps=ph,
                                                                             full_dist=False,
                                                                             return_mode_prob=True)

                        for ts in predictions:
                            for node in predictions[ts]:
                                instance_token = node.id
                                sample_token = scene.sample_tokens[ts]
                                # get the ground truth
                                ground_truth = node.get(np.array([ts+1, ts+ph]),  {'position':['x','y']}) + np.array([scene.x_min, scene.y_min])
                                ns_pred = Prediction(instance=instance_token,
                                                     sample=sample_token,
                                                     prediction=predictions[ts][node][0] + np.array([scene.x_min, scene.y_min]),
                                                     probabilities=mode_probs[ts][node][0,:,0])
                                metric_pred = {}
                                for metric in ns_metrics:
                                    metric_val = metric['cls'](ground_truth, ns_pred)[0]
                                    for k, val in zip(k_to_report, metric_val):
                                        k = str(k)
                                        metric_pred[metric['name']+k] = [val]
                                eval_batch_errors.append({eval_env.NodeType.VEHICLE: metric_pred})

                    evaluation.log_batch_errors(eval_batch_errors,
                                                log_writer,
                                                'eval',
                                                epoch)

                elif hyperparams['eval_type'] == "trajectron":
                    eval_batch_errors = []
                    for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                        timesteps = scene.sample_timesteps(hyperparams['eval_batch_size'])

                        predictions = eval_trajectron.predict(scene,
                                                              timesteps,
                                                              ph,
                                                              num_samples=50,
                                                              min_future_timesteps=ph,
                                                              full_dist=False)

                        eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                     scene.dt,
                                                                                     max_hl=max_hl,
                                                                                     ph=ph,
                                                                                     node_type_enum=eval_env.NodeType,
                                                                                     map=scene.map))

                    evaluation.log_batch_errors(eval_batch_errors,
                                                log_writer,
                                                'eval',
                                                epoch,
                                                bar_plot=['kde'],
                                                box_plot=['ade', 'fde'])

                eval_batch_errors_ml = []
                for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(scene.timesteps)

                    predictions = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=1,
                                                          min_future_timesteps=ph,
                                                          z_mode=True,
                                                          gmm_mode=True,
                                                          full_dist=False)

                    eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                                                                                    scene.dt,
                                                                                    max_hl=max_hl,
                                                                                    ph=ph,
                                                                                    map=scene.map,
                                                                                    node_type_enum=eval_env.NodeType,
                                                                                    kde=False))

                evaluation.log_batch_errors(eval_batch_errors_ml,
                                            log_writer,
                                            'eval/ml',
                                            epoch)



        if hyperparams['save_every'] is not None and hyperparams['debug'] is False and epoch % hyperparams['save_every'] == 0:
            model_registrar.save_models(epoch)
            with open(os.path.join(model_dir, 'curr_iter_%d.json' % epoch), 'w') as F:
                json.dump({k.name:v for k,v in curr_iter_node_type.items()}, F)
            annealers_state_dict = {}
            optim_state_dict = {}
            optim_d_state_dict = {}
            lr_scheduler_state_dict = {}
            lr_scheduler_d_state_dict = {}
            for node_type in trajectron.node_models_dict:
                annealers_state_dict[node_type.name] = {}
                node_model = trajectron.node_models_dict[node_type]
                for name, schedul in zip(node_model.annealed_vars, node_model.schedulers):
                    annealers_state_dict[node_type][name] = schedul.state_dict()
                lr_scheduler_state_dict[node_type.name] = lr_scheduler[node_type].state_dict()
                lr_scheduler_d_state_dict[node_type.name] = lr_scheduler_d[node_type].state_dict()
                optim_state_dict[node_type.name] = optimizer[node_type].state_dict()
                optim_d_state_dict[node_type.name] = optimizer_d[node_type].state_dict()
            torch.save(annealers_state_dict, os.path.join(model_dir, 'annealers_%d.pt' % epoch))
            torch.save(lr_scheduler_state_dict, os.path.join(model_dir, 'lr_scheduler_%d.pt' % epoch))
            torch.save(lr_scheduler_d_state_dict, os.path.join(model_dir, 'lr_scheduler_d_%d.pt' % epoch))
            torch.save(optim_state_dict, os.path.join(model_dir, 'optim_%d.pt' % epoch))
            torch.save(optim_d_state_dict, os.path.join(model_dir, 'optim_d_%d.pt' % epoch))


if __name__ == '__main__':
    main()
