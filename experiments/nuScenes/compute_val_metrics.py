import sys
import pickle
import argparse

from pathlib import Path
from collections import defaultdict

nu_path = '/root/workspace/nuscenes-devkit/python-sdk/'
sys.path.insert(0,nu_path)
sys.path.append("../../trajectron")

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.eval.prediction.compute_metrics import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--checkpoint', type=int, required=True)
args = parser.parse_args()
if __name__ == "__main__":
    dataroot = "/datasets_local/nuscenes"
    version = "v1.0-trainval"
    config_name = "predict_2020_icra.json"
    model_dir = Path(args.model_dir)
    pkls_dir = model_dir.joinpath("metrics","epoch,%d" % args.checkpoint, "val_open_loop")
    if not pkls_dir.exists():
        pkls_dir = model_dir.joinpath("metrics","epoch,%d" % args.checkpoint, "val")
    metric_path = pkls_dir.joinpath('metrics.pkl')
    if not metric_path.exists():
        nusc = NuScenes(version=version, dataroot=dataroot)
        helper = PredictHelper(nusc)
        config = load_prediction_config(helper, config_name)
        split = get_prediction_challenge_split("val", dataroot=dataroot)
        predictions = []
        prediction_instances = set()
        for pkl_path in pkls_dir.iterdir():
            with open(pkl_path, "rb") as F:
                pred = pickle.load(F)
                for (instance, sample), pred_dict in pred.items():
                    instance_sample = instance+'_'+sample
                    if ('mode-pred@6s' in pred_dict) and (instance_sample in split) and (instance_sample not in prediction_instances):
                        predictions.append(pred_dict['mode-pred@6s'])
                        prediction_instances.add(instance_sample)
        metrics = compute_metrics(predictions, helper, config)
        metrics['missing_instances'] = len(set(split) - prediction_instances)
        with open(metric_path, "wb") as f:
            pickle.dump(metrics, f)
    else:
        with open(metric_path, "rb") as f:
            metrics = pickle.load(f)

    k_to_report = [1,5,10]
    for metric_name in ['MinFDEK','MinADEK']:
        for k_id, k in enumerate(k_to_report):
            metric_val = metrics[metric_name]['RowMean'][k_id]
            print(metric_name + str(k) + ": %1.2f" % metric_val)
    for metric_name, row_idx in zip(['MissRateTopK_2', "OffRoadRate"], [1,0]):
        metric_val = metrics[metric_name]['RowMean'][row_idx]
        print(metric_name + ": %1.2f" % metric_val)
