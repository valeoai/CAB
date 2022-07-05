import sys
import os
import dill
import json
# import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline

from pathlib import Path
import sys
nu_path = '/root/workspace/nuscenes-devkit/python-sdk/'
sys.path.insert(0,nu_path)
from nuscenes.eval.prediction.data_classes import Prediction
import pickle
from options import Options, OptionsDict, deep_asdict

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def compute_road_violations(predicted_trajs, map, channel):
    obs_map = 1 - map.data[..., channel, :, :] / 255

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[0]),
                                         range(obs_map.shape[1]),
                                         obs_map,
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1], old_shape[2]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=2) > 0, dtype=float)

    return num_viol_trajs


def load_model(model_dir, env, ts=100, device="cuda:0"):
    if not Path(model_dir).joinpath('model_registrar-%d.pt' % ts).exists():
        model_dir = list(Path(model_dir).glob("resume,*"))[-1]
        model_dir = str(model_dir)
      
    model_registrar = ModelRegistrar(model_dir, device)

    model_registrar.load_models(ts)
    hyperparams = deep_asdict(Options().options)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    data_path = os.path.join(Options()['data_dir'], Options()['eval_data_dict'])
    with open(data_path, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    device = "cuda:0"

    model_dir = os.path.join(Options()['log_dir'], Options()['ts_tag'])
    eval_stg, hyperparams = load_model(model_dir, env, ts=int(Options()['resume']), device=device)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    max_hl = hyperparams['maximum_history_length']

    exp_dir = Path(model_dir)
    metrics_dir = exp_dir.joinpath('metrics')
    metrics_dir.mkdir(exist_ok=True)
    if hyperparams['eval_branch'] == "state":
        metrics_dir = metrics_dir.joinpath("epoch,%d_state"%int(hyperparams['resume']))
        branch = "state"
    else:
        metrics_dir = metrics_dir.joinpath("epoch,%d"%int(hyperparams['resume']))
        branch = "full"
    metrics_dir.mkdir(exist_ok=True)

    split = Path(data_path).stem.replace('nuScenes_','').replace('_full','')


    preds_dir = metrics_dir.joinpath(split)
    preds_dir.mkdir(exist_ok=True)
    if 'eval_type' not in hyperparams:
        hyperparams['eval_type'] = "nuscenes"

    ph = 12
    for scene in tqdm(scenes, desc='Evaluation', ncols=80):
        eval_batch_errors = {}
        timesteps = scene.sample_timesteps(scene.timesteps)
        with torch.no_grad():
            mode_probs, predictions = eval_stg.predict(scene,
                                                        timesteps,
                                                        ph,
                                                        branch=branch,
                                                        num_samples=1,
                                                        all_z_sep=True,
                                                        gmm_mode=True,
                                                        min_future_timesteps=ph,
                                                        full_dist=False,
                                                        return_mode_prob=True)
        for ts in predictions:
            for node in predictions[ts]:
                instance_token = node.id
                sample_token = scene.sample_tokens[ts]
                ns_pred = Prediction(instance=instance_token,
                                        sample=sample_token,
                                        prediction=predictions[ts][node][0] + np.array([scene.x_min, scene.y_min]),
                                        probabilities=mode_probs[ts][node][0,:,0])
                if (instance_token, sample_token) not in eval_batch_errors:
                    eval_batch_errors[(instance_token, sample_token)] = {}
                eval_batch_errors[(instance_token, sample_token)].update({
                                                                        "mode-pred@%ds" % (scene.dt*ph) :ns_pred.serialize()
                                                                        })
        with open(preds_dir.joinpath(f"{split}-{scene.name}.pkl"), 'wb') as F:
            pickle.dump(eval_batch_errors, F)

