import torch
import numpy as np

from model.dataset import get_timesteps_data, restore
from model.mgcvae import MultimodalGenerativeCVAE
from model.mgcvae_rubiz import MultimodalGenerativeCVAERubiZ
from model.mgcvae_reweight import MultimodalGenerativeCVAEReweight
from model.mgcvae_cab import MultimodalGenerativeCVAECAB
from model.mgcvae_halentnet import MultimodalGenerativeCVAEHalentNet


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                if "cab_params" in self.hyperparams:
                    self.node_models_dict[node_type] = MultimodalGenerativeCVAECAB(env,
                                                                                    node_type,
                                                                                    self.model_registrar,
                                                                                    self.hyperparams,
                                                                                    self.device,
                                                                                    edge_types,
                                                                                    log_writer=self.log_writer)
                elif "rubiz_params" in self.hyperparams:
                    self.node_models_dict[node_type] = MultimodalGenerativeCVAERubiZ(env,
                                                                                    node_type,
                                                                                    self.model_registrar,
                                                                                    self.hyperparams,
                                                                                    self.device,
                                                                                    edge_types,
                                                                                    log_writer=self.log_writer)
                elif "reweight_params" in self.hyperparams:
                    self.node_models_dict[node_type] = MultimodalGenerativeCVAEReweight(env,
                                                                                    node_type,
                                                                                    self.model_registrar,
                                                                                    self.hyperparams,
                                                                                    self.device,
                                                                                    edge_types,
                                                                                    log_writer=self.log_writer)
                elif "halentnet" in self.hyperparams:
                    self.node_models_dict[node_type] = MultimodalGenerativeCVAEHalentNet(env,
                                                                                    node_type,
                                                                                    self.model_registrar,
                                                                                    self.hyperparams,
                                                                                    self.device,
                                                                                    edge_types,
                                                                                    log_writer=self.log_writer)
                else:
                    self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                                node_type,
                                                                                self.model_registrar,
                                                                                self.hyperparams,
                                                                                self.device,
                                                                                edge_types,
                                                                                log_writer=self.log_writer)

    def set_log_writer(self, log_writer):
        self.log_writer = log_writer
        for node_model in self.node_models_dict.values():
            node_model.log_writer = log_writer

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def train_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph)

        return loss

    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def predict(self,
                scene,
                timesteps,
                ph,
                branch="full",
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                ego_only=False,
                return_mode_prob=False,
                query_node=None):

        predictions_dict = {}
        batches = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams, ego_only=ego_only, query_node=query_node)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch


            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            if isinstance(model, (MultimodalGenerativeCVAERubiZ, MultimodalGenerativeCVAECAB)):
                predictions = model.predict(inputs=x,
                                            inputs_st=x_st_t,
                                            first_history_indices=first_history_index,
                                            neighbors=neighbors_data_st,
                                            neighbors_edge_value=neighbors_edge_value,
                                            robot=robot_traj_st_t,
                                            map=map,
                                            prediction_horizon=ph,
                                            num_samples=num_samples,
                                            branch=branch,
                                            z_mode=z_mode,
                                            gmm_mode=gmm_mode,
                                            full_dist=full_dist,
                                            all_z_sep=all_z_sep,
                                            return_mode_prob=return_mode_prob)
            else:
                predictions = model.predict(inputs=x,
                                            inputs_st=x_st_t,
                                            first_history_indices=first_history_index,
                                            neighbors=neighbors_data_st,
                                            neighbors_edge_value=neighbors_edge_value,
                                            robot=robot_traj_st_t,
                                            map=map,
                                            prediction_horizon=ph,
                                            num_samples=num_samples,
                                            z_mode=z_mode,
                                            gmm_mode=gmm_mode,
                                            full_dist=full_dist,
                                            all_z_sep=all_z_sep,
                                            return_mode_prob=return_mode_prob)

            if return_mode_prob:
                mode_prob, predictions = predictions
                mode_prob_dict = {}
                mode_prob_np = mode_prob.cpu().detach().numpy()
            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
                if return_mode_prob:
                    if ts not in mode_prob_dict.keys():
                        mode_prob_dict[ts] = dict()
                    mode_prob_dict[ts][nodes[i]] = np.transpose(mode_prob_np[[i],:], (1, 2, 0))
            batches[node_type] = batch

        if return_mode_prob:
            return mode_prob_dict, predictions_dict
        else:
            return predictions_dict

    def bias_full_kl(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                ego_only=False,
                query_node=None):

        full_dist = True
        z_mode=False
        gmm_mode=False
        all_z_sep=False

        kl_sf_dict = {}
        kl_fs_dict = {}
        batches = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]
            assert isinstance(model, (MultimodalGenerativeCVAERelativeDifficulty, 
                                      MultimodalGenerativeCVAEParallelBaseline,
                                      MultimodalGenerativeCVAERubiZ))

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams, ego_only=ego_only, query_node=query_node)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch


            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            kl_sf, kl_fs = model.bias_full_kl(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            kl_sf_np = kl_sf.cpu().detach().numpy()
            kl_fs_np = kl_fs.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in kl_sf_dict.keys():
                    kl_sf_dict[ts] = dict()
                    kl_fs_dict[ts] = dict()
                kl_sf_dict[ts][nodes[i]] = kl_sf_np[:, i]
                kl_fs_dict[ts][nodes[i]] = kl_fs_np[:, i]
            batches[node_type] = batch

        return kl_sf_dict, kl_fs_dict

    def gan_d_loss(self, batch, node_type, grid_std=3, grid_max=9):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        d_loss = model.gan_dc_loss(inputs=x,
                                   inputs_st=x_st_t,
                                   first_history_indices=first_history_index,
                                   labels=y,
                                   labels_st=y_st_t,
                                   neighbors=restore(neighbors_data_st),
                                   neighbors_edge_value=restore(neighbors_edge_value),
                                   robot=robot_traj_st_t,
                                   map=map,
                                   prediction_horizon=self.ph,
                                   grid_std=grid_std, grid_max=grid_max)

        return d_loss

    def gan_g_loss(self, batch, node_type, grid_std=3, grid_max=9, creative=None):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        g_loss, gc_loss = model.gan_gc_loss(inputs=x,
                                            inputs_st=x_st_t,
                                            first_history_indices=first_history_index,
                                            labels=y,
                                            labels_st=y_st_t,
                                            neighbors=restore(neighbors_data_st),
                                            neighbors_edge_value=restore(neighbors_edge_value),
                                            robot=robot_traj_st_t,
                                            map=map,
                                            prediction_horizon=self.ph,
                                            grid_std=grid_std, grid_max=grid_max,
                                            creative=creative)

        return g_loss, gc_loss

    def visualize_gan(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        our_sampled_future, labels, map, rotate_f = model.visualization_gan(inputs=x,
                                                                            inputs_st=x_st_t,
                                                                            first_history_indices=first_history_index,
                                                                            labels=y,
                                                                            labels_st=y_st_t,
                                                                            neighbors=restore(neighbors_data_st),
                                                                            neighbors_edge_value=restore(
                                                                                neighbors_edge_value),
                                                                            robot=robot_traj_st_t,
                                                                            map=map,
                                                                            prediction_horizon=self.ph)

        return our_sampled_future, labels, map, rotate_f


    def get_shapley_values(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                ego_only=False,
                query_node=None):

        shapleys_dict = {}
        batches = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams, ego_only=ego_only, query_node=query_node)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch


            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            shapleys = model.get_shapley_values(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples)

            shapleys_np = shapleys.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in shapleys_dict.keys():
                    shapleys_dict[ts] = dict()
                shapleys_dict[ts][nodes[i]] = shapleys_np[i]
            batches[node_type] = batch

        return shapleys_dict