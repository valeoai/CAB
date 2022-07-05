import warnings

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model.dynamics as dynamic_module

from environment.scene_graph import DirectedEdge

from model.components import *
from model.model_utils import *


class MultimodalGenerativeCVAERubiZ(object):
    def __init__(self,
                 env,
                 node_type,
                 model_registrar,
                 hyperparams,
                 device,
                 edge_types,
                 log_writer=None):
        self.hyperparams = hyperparams
        self.env = env
        self.node_type = node_type
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = [edge_type for edge_type in edge_types if edge_type[0] is node_type]
        self.curr_iter = 0

        self.node_modules = dict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type]
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
        if self.hyperparams['incl_robot_node']:
            self.robot_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[env.robot_type].values()])
            )
        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))

        edge_types_str = [DirectedEdge.get_str_from_types(*edge_type) for edge_type in self.edge_types]
        self.create_graphical_model(edge_types_str)

        dynamic_class = getattr(dynamic_module, hyperparams['dynamic'][self.node_type]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        ####
        # RUBIZ
        self.dynamic_full = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type, branch="full_branch")
        self.dynamic_state = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_so_size, self.node_type, branch="state_branch")
        self.control_dist = {}
        ####

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(self.node_type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule(self.node_type + '/node_future_encoder',
                           model_if_absent=nn.LSTM(input_size=self.pred_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule(self.node_type + '/node_future_encoder/initial_h',
                           model_if_absent=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule(self.node_type + '/node_future_encoder/initial_c',
                           model_if_absent=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))

        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        if self.hyperparams['incl_robot_node']:
            self.add_submodule('robot_future_encoder',
                               model_if_absent=nn.LSTM(input_size=self.robot_state_length,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                       bidirectional=True,
                                                       batch_first=True))
            # These are related to how you initialize states for the robot future encoder.
            self.add_submodule('robot_future_encoder/initial_h',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))
            self.add_submodule('robot_future_encoder/initial_c',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))

        if self.hyperparams['edge_encoding']:
            ##############################
            #   Edge Influence Encoder   #
            ##############################
            # NOTE: The edge influence encoding happens during calls
            # to forward or incremental_forward, so we don't create
            # a model for it here for the max and sum variants.
            if self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
                                                           hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
                                                           bidirectional=True,
                                                           batch_first=True))

                # Four times because we're trying to mimic a bi-directional
                # LSTM's output (which, here, is c and h from both ends).
                self.eie_output_dims = 4 * self.hyperparams['enc_rnn_dim_edge_influence']

            elif self.hyperparams['edge_influence_combine_method'] == 'attention':
                # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
                # We calculate an attention context vector using the encoded edges as the "encoder"
                # (that we attend _over_)
                # and the node history encoder representation as the "decoder state" (that we attend _on_).
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=AdditiveAttention(
                                       encoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_edge_influence'],
                                       decoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_history']))

                self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']

        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams['use_map_encoding']:
            if self.node_type in self.hyperparams['map_encoder']:
                me_params = self.hyperparams['map_encoder'][self.node_type]
                self.add_submodule(self.node_type + '/map_encoder',
                                   model_if_absent=CNNMapEncoder(me_params['map_channels'],
                                                                 me_params['hidden_channels'],
                                                                 me_params['output_size'],
                                                                 me_params['masks'],
                                                                 me_params['strides'],
                                                                 me_params['patch_size']))

        ################################
        #   Discrete Latent Variable   #
        ################################
        #####
        # RUBIZ
        self.latent_full = DiscreteLatent(self.hyperparams, self.device)
        self.latent_state = DiscreteLatent(self.hyperparams, self.device)
        self.latent_fused = DiscreteLatent(self.hyperparams, self.device)
        #####

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']
        x_so_size = self.hyperparams['enc_rnn_dim_history'] # RUBIZ
        if self.hyperparams['edge_encoding']:
            #              Edge Encoder
            x_size += self.eie_output_dims
        if self.hyperparams['incl_robot_node']:
            #              Future Conditional Encoder
            x_size += 4 * self.hyperparams['enc_rnn_dim_future']
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            #              Map Encoder
            x_size += self.hyperparams['map_encoder'][self.node_type]['output_size']

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            self.add_submodule(self.node_type + '/so' + '/p_z_x',
                               model_if_absent=nn.Linear(x_so_size,
                                                         self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']

            self.add_submodule(self.node_type + '/so' + '/hx_to_z',
                               model_if_absent=nn.Linear(hx_size, self.latent_full.z_dim))
        else:
            hx_size = x_size
            self.add_submodule(self.node_type + '/so' + '/hx_to_z',
                               model_if_absent=nn.Linear(x_so_size, self.latent_state.z_dim))

        self.add_submodule(self.node_type + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent_full.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/q_z_xy',
                               #                                           Node Future Encoder
                               model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            self.add_submodule(self.node_type + '/srbanch' + '/q_z_xy',
                               #                                           Node Future Encoder
                               model_if_absent=nn.Linear(x_so_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule(self.node_type + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent_full.z_dim))
        self.add_submodule(self.node_type + '/so' + '/hxy_to_z',
                           model_if_absent=nn.Linear(x_so_size + 4 * self.hyperparams['enc_rnn_dim_future'], self.latent_state.z_dim))

        ####################
        #   Decoder LSTM   #
        ####################
        if self.hyperparams['incl_robot_node']:
            decoder_input_dims = self.pred_state_length + self.robot_state_length + z_size + x_size
        else:
            decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule(self.node_type + '/decoder/state_action',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))

        self.add_submodule(self.node_type + '/decoder/rnn_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))


        self.add_submodule(self.node_type + '/so' + '/decoder/state_action',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))
        self.add_submodule(self.node_type + '/so' + '/decoder/rnn_cell',
                           model_if_absent=nn.GRUCell(self.pred_state_length + z_size + x_so_size,
                                                      self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/so' + '/decoder/initial_h',
                           model_if_absent=nn.Linear(z_size + x_so_size,
                                                     self.hyperparams['dec_rnn_dim']))

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_pis',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_sigmas',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_corrs',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))

        self.add_submodule(self.node_type + '/so' + '/decoder/proj_to_GMM_log_pis',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))
        self.add_submodule(self.node_type + '/so' + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/so' + '/decoder/proj_to_GMM_log_sigmas',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/so' + '/decoder/proj_to_GMM_corrs',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))


        ###########################
        #   Rubi Distrib fusion   #
        ###########################
        self.add_submodule(self.node_type + '/decoder/rubi/proj_to_corrs',
                           model_if_absent=nn.Linear(3,1))
        self.add_submodule(self.node_type + '/decoder/rubi/proj_to_sigmas',
                           model_if_absent=nn.Linear(3,self.pred_state_length))

        if self.hyperparams['rubiz_params'].get('use_fusion_proj', False):
            self.add_submodule(self.node_type + '/rubiz_fusion_p',
                               model_if_absent=nn.Linear(z_size,z_size))
            self.add_submodule(self.node_type + '/rubiz_fusion_q',
                               model_if_absent=nn.Linear(z_size,z_size))

        self.x_size = x_size
        self.x_so_size = x_so_size
        self.z_size = z_size

    def create_edge_models(self, edge_types):
        for edge_type in edge_types:
            neighbor_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[edge_type.split('->')[1]].values()]))
            if self.hyperparams['edge_state_combine_method'] == 'pointnet':
                self.add_submodule(edge_type + '/pointnet_encoder',
                                   model_if_absent=nn.Sequential(
                                       nn.Linear(self.state_length, 2 * self.state_length),
                                       nn.ReLU(),
                                       nn.Linear(2 * self.state_length, 2 * self.state_length),
                                       nn.ReLU()))

                edge_encoder_input_size = 2 * self.state_length + self.state_length

            elif self.hyperparams['edge_state_combine_method'] == 'attention':
                self.add_submodule(self.node_type + '/edge_attention_combine',
                                   model_if_absent=TemporallyBatchedAdditiveAttention(
                                       encoder_hidden_state_dim=self.state_length,
                                       decoder_hidden_state_dim=self.state_length))
                edge_encoder_input_size = self.state_length + neighbor_state_length

            else:
                edge_encoder_input_size = self.state_length + neighbor_state_length

            self.add_submodule(edge_type + '/edge_encoder',
                               model_if_absent=nn.LSTM(input_size=edge_encoder_input_size,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_edge'],
                                                       batch_first=True))

    def create_graphical_model(self, edge_types):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        #####################
        #   Edge Encoders   #
        #####################
        if self.hyperparams['edge_encoding']:
            self.create_edge_models(edge_types)

        for name, module in self.node_modules.items():
            module.to(self.device)

    def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            annealer_kws['device'] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
            rsetattr(self, name + '_optimizer', dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer,
                                       value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(name='kl_weight',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['kl_weight_start'],
                                      'finish': self.hyperparams['kl_weight'],
                                      'center_step': self.hyperparams['kl_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
                                          'kl_sigmoid_divisor']
                                  })

        self.create_new_scheduler(name='latent_full.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent_full.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])

        self.create_new_scheduler(name='latent_state.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent_state.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])

        self.create_new_scheduler(name='latent_fused.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent_fused.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])
    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])

        self.summarize_annealers()

    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.add_scalar('%s/%s' % (str(self.node_type), annealed_var.replace('.', '/')),
                                               rgetattr(self, annealed_var), self.curr_iter)

    def obtain_encoded_tensors(self,
                               mode,
                               inputs,
                               inputs_st,
                               labels,
                               labels_st,
                               first_history_indices,
                               neighbors,
                               neighbors_edge_value,
                               robot,
                               map) -> (torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor):
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        x, x_r_t, y_e, y_r, y = None, None, None, None, None
        initial_dynamics = dict()

        batch_size = inputs.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history = inputs
        node_present_state = inputs[:, -1]
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]

        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]
        node_pos_st = inputs_st[:, -1, 0:2]
        node_vel_st = inputs_st[:, -1, 2:4]

        n_s_t0 = node_present_state_st

        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel


        self.dynamic_state.set_initial_condition(initial_dynamics)
        self.dynamic_full.set_initial_condition(initial_dynamics)

        if self.hyperparams['incl_robot_node']:
            x_r_t, y_r = robot[..., 0, :], robot[..., 1:, :]

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st,
                                                        first_history_indices)
        x_so = node_history_encoded


        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = labels_st

        ##############################
        # Encode Node Edges per Type #
        ##############################
        if self.hyperparams['edge_encoding']:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                # Encode edges for given edge type
                encoded_edges_type = self.encode_edge(mode,
                                                      node_history,
                                                      node_history_st,
                                                      edge_type,
                                                      neighbors[edge_type],
                                                      neighbors_edge_value[edge_type],
                                                      first_history_indices)
                node_edges_encoded.append(encoded_edges_type)  # List of [bs/nbs, enc_rnn_dim]
            #####################
            # Encode Node Edges #
            #####################
            total_edge_influence = self.encode_total_edge_influence(mode,
                                                                    node_edges_encoded,
                                                                    node_history_encoded,
                                                                    batch_size)

        ################
        # Map Encoding #
        ################
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            if self.log_writer and (self.curr_iter + 1) % 500 == 0:
                map_clone = map.clone()
                map_patch = self.hyperparams['map_encoder'][self.node_type]['patch_size']
                map_clone[:, :, map_patch[1] - 5:map_patch[1] + 5, map_patch[0] - 5:map_patch[0] + 5] = 1.
                self.log_writer.add_images(f"{self.node_type}/cropped_maps", map_clone,
                                           self.curr_iter, dataformats='NCWH')

            encoded_map = self.node_modules[self.node_type + '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
            do = self.hyperparams['map_encoder'][self.node_type]['dropout']
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        x_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams['edge_encoding']:
            x_concat_list.append(total_edge_influence)  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        # HEDI: the input state dropout is done here and not before the edge encoding, as this info serves as context for
        # attention.
        if (mode == ModeKeys.TRAIN) and (self.node_type=="VEHICLE") and ("statedropout_keep_prob" in self.hyperparams):
            p = self.hyperparams['statedropout_keep_prob']
            statedropout_mask = torch.bernoulli(torch.ones(batch_size)*p)
            statedropout_mask = statedropout_mask.to(self.device)
            node_history_encoded = node_history_encoded*statedropout_mask[:,None]

        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams['incl_robot_node']:
            robot_future_encoder = self.encode_robot_future(mode, x_r_t, y_r)
            x_concat_list.append(robot_future_encoder)

        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            if self.log_writer:
                self.log_writer.add_scalar(f"{self.node_type}/encoded_map_max",
                                           torch.max(torch.abs(encoded_map)), self.curr_iter)
            x_concat_list.append(encoded_map)

        x = torch.cat(x_concat_list, dim=1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y)

        # HEDI:
        # x: node features, containing history, edges, robot_future and map
        # x_r_t: robot features
        # y_e: encoding of future node traj (used in training of the cvae)
        # y_r: encoding of the future robot
        # y: future pred_state (=position) of the node
        # n_s_t0: current state of the node
        return x, x_so, x_r_t, y_e, y_r, y, n_s_t0

    def encode_node_history(self, mode, node_hist, first_history_indices):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.node_type + '/node_history_encoder'],
                                                      original_seqs=node_hist,
                                                      lower_indices=first_history_indices)

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]


        last_index_per_sequence = -(first_history_indices + 1)
        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_edge(self,
                    mode,
                    node_history,
                    node_history_st,
                    edge_type,
                    neighbors,
                    neighbors_edge_value,
                    first_history_indices):

        max_hl = self.hyperparams['maximum_history_length']

        edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
        for i, neighbor_states in enumerate(neighbors):  # Get neighbors for timestep in batch
            if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
                )
                edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
            else:
                edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))

        if self.hyperparams['edge_state_combine_method'] == 'sum':
            # Used in Structural-RNN to combine edges as well.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.to(self.device),
                                                                           dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams['edge_state_combine_method'] == 'max':
            # Used in NLP, e.g. max over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.max(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.max(edge_value.to(self.device),
                                                                           dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams['edge_state_combine_method'] == 'mean':
            # Used in NLP, e.g. mean over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.mean(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.mean(edge_value.to(self.device),
                                                                            dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1) #HEDI: node history + local aggregated neighbors history

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + '/edge_encoder'],
            original_seqs=joint_history,
            lower_indices=first_history_indices
        )

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        ret = outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
        if self.hyperparams['dynamic_edges'] == 'yes':
            return ret * combined_edge_masks
        else:
            return ret

    def encode_total_edge_influence(self, mode, encoded_edges, node_history_encoder, batch_size):
        if self.hyperparams['edge_influence_combine_method'] == 'sum':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.sum(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'mean':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.mean(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'max':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.max(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)

                _, state = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges)
                combined_edges = unpack_RNN_state(state)
                combined_edges = F.dropout(combined_edges,
                                           p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                           training=(mode == ModeKeys.TRAIN))

        elif self.hyperparams['edge_influence_combine_method'] == 'attention':
            # Used in Social Attention (https://arxiv.org/abs/1710.04689)
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)
                combined_edges, _ = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges,
                                                                                                  node_history_encoder)
                combined_edges = F.dropout(combined_edges,
                                           p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                           training=(mode == ModeKeys.TRAIN))
                # HEDI: an attention over the different types of edges, with node_history_encoder as context

        return combined_edges

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules[self.node_type + '/node_future_encoder/initial_h']
        initial_c_model = self.node_modules[self.node_type + '/node_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules[self.node_type + '/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def encode_robot_future(self, mode, robot_present, robot_future) -> torch.Tensor:
        """
        Encodes the robot future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param robot_present: Current state of the robot. [bs, state]
        :param robot_future: Future states of the robot. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules['robot_future_encoder/initial_h']
        initial_c_model = self.node_modules['robot_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(robot_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(robot_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules['robot_future_encoder'](robot_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def q_z_xy(self, mode, x, y_e, branch) -> torch.Tensor:
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([x, y_e], dim=1)
        if branch == "state":
            branch_txt = "/so"
            latent = self.latent_state
        else:
            branch_txt = ""
            latent = self.latent_full

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules[self.node_type + branch_txt + '/q_z_xy']
            h = F.dropout(F.relu(dense(xy)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules[self.node_type + branch_txt + '/hxy_to_z']
        return latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, mode, x, branch):
        r"""
        .. math:: p_\theta(z \mid \mathbf{x}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :return: Latent distribution of the CVAE.
        """
        if branch == "state":
            branch_txt = "/so"
            latent = self.latent_state
        else:
            branch_txt = ""
            latent = self.latent_full

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules[self.node_type + branch_txt + '/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules[self.node_type + '/hx_to_z']
        return latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_params(self, tensor, branch_txt) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.node_modules[self.node_type + branch_txt + '/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules[self.node_type + branch_txt +'/decoder/proj_to_GMM_mus'](tensor)
        log_sigmas = self.node_modules[self.node_type + branch_txt +'/decoder/proj_to_GMM_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules[self.node_type + branch_txt +'/decoder/proj_to_GMM_corrs'](tensor))
        return log_pis, mus, log_sigmas, corrs

    def p_y_xz(self, mode, x, x_nr_t, y_r, n_s_t0, z_stacked, prediction_horizon,
               num_samples, branch, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """

        ph = prediction_horizon
        pred_dim = self.pred_state_length

        if branch == "full":
            branch_txt = ""
            latent = self.latent_full
            dynamic = self.dynamic_full
        elif branch == "state":
            branch_txt = "/so"
            latent = self.latent_state
            dynamic = self.dynamic_state
        elif branch == "fused":
            branch_txt = ""
            latent = self.latent_fused
            dynamic = self.dynamic_full
        else:
            raise ValueError(branch)

        z = torch.reshape(z_stacked, (-1, latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.node_type + branch_txt + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + branch_txt + '/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type + branch_txt + '/decoder/state_action'](n_s_t0)

        state = initial_state
        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([zx,
                                a_0.repeat(num_samples * num_components, 1),
                                x_nr_t.repeat(num_samples * num_components, 1)], dim=1)
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)
            # HEDI: for each batch element, input_ contains the current position and its associated node
            # feature for each dimension of latent space (categorical one hot)

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state, branch_txt)

            # debugging
            if torch.isnan(log_pi_t).max() + torch.isinf(log_pi_t).max():
                import ipdb; ipdb.set_trace()
            if torch.isnan(mu_t).max() + torch.isinf(mu_t).max():
                import ipdb; ipdb.set_trace()
            if torch.isnan(log_sigma_t).max() + torch.isinf(log_sigma_t).max():
                import ipdb; ipdb.set_trace()
            if torch.isnan(corr_t).max() + torch.isinf(corr_t).max():
                import ipdb; ipdb.set_trace()

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components) # HEDI: bsize x 2*num_components
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, num_components))

            if self.hyperparams['incl_robot_node']:
                dec_inputs = [zx, a_t, y_r[:, j].repeat(num_samples * num_components, 1)]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state


        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        # debug
        if torch.isnan(corrs).max() + torch.isinf(corrs).max():
            import ipdb; ipdb.set_trace()

        # HEDI: gaussian mixture with 25 modes (in z), each with proba stored in log_pis.
        a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                       torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        self.control_dist[branch] = a_dist


        if self.hyperparams['dynamic'][self.node_type]['distribution']:
            # HEDI: a_dist is a distribution over CONTROL values: omega and accel (for each timestep, for each component in z).
            y_dist = dynamic.integrate_distribution(a_dist, x)
        else:
            # HEDI: a_dist is a distribution over positions (x,y).
            y_dist = a_dist
        # HEDI: y_dist is a distribution over positions (x,y)

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            sampled_future = dynamic.integrate_samples(a_sample, x) #HEDI: if a_dist is the position distribution, why do we integrate ?
            return y_dist, sampled_future
        else:
            return y_dist

    def encoder(self, mode, x, y_e, branch, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams['k']
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams['k_eval']
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        if branch == "full":
            latent = self.latent_full
        elif branch == "state":
            latent = self.latent_state
        else:
            raise ValueError(branch)

        latent.q_dist = self.q_z_xy(mode, x, y_e, branch)
        latent.p_dist = self.p_z_x(mode, x, branch)

        z = latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = latent.kl_q_p(self.log_writer, '%s/%s' % (str(self.node_type),branch), self.curr_iter)
            if self.log_writer is not None:
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'kl_%s'%branch), kl_obj, self.curr_iter)
        else:
            kl_obj = None

        return z, kl_obj


    def decoder(self, mode, x, x_so, x_nr_t, y, y_r, n_s_t0, z, z_so, labels, prediction_horizon, num_samples):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        num_components = self.hyperparams['N'] * self.hyperparams['K']

        y_dist_so = self.p_y_xz(mode, x_so, x_nr_t, y_r, n_s_t0, z_so,
                             prediction_horizon, num_samples, "state", num_components=num_components)

        y_dist = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                             prediction_horizon, num_samples, "full", num_components=num_components)

        y_dist_fused = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                             prediction_horizon, num_samples, "fused", num_components=num_components)


        log_p_yt_xz_so = torch.clamp(y_dist_so.log_prob(labels),
                                          min=-self.hyperparams['log_p_yt_xz_max'],
                                          max=self.hyperparams['log_p_yt_xz_max'])
        log_p_yt_xz_fused = torch.clamp(y_dist_fused.log_prob(labels),
                                        min=-self.hyperparams['log_p_yt_xz_max'],
                                        max=self.hyperparams['log_p_yt_xz_max'])
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels),
                                        min=-self.hyperparams['log_p_yt_xz_max'],
                                        max=self.hyperparams['log_p_yt_xz_max'])

        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_yt_xz_so'), log_p_yt_xz_so, self.curr_iter)
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_yt_xz_full'), log_p_yt_xz, self.curr_iter)
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_yt_xz_fused'), log_p_yt_xz_fused, self.curr_iter)

        log_p_y_xz_fused = torch.sum(log_p_yt_xz_fused, dim=2)
        log_p_y_xz_so = torch.sum(log_p_yt_xz_so, dim=2)
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)

        return log_p_y_xz_fused, log_p_y_xz_so, log_p_y_xz

    def fuse_latents(self, mode):
        latent_proj_p =  self.node_modules.get(self.node_type + "/rubiz_fusion_p", None)
        so_logits_p = self.latent_state.p_dist.logits
        if latent_proj_p is not None:
            so_logits_p = latent_proj_p(so_logits_p)

        if self.hyperparams['rubiz_params']['fusion_type'] == 'sum_of_logits':
            new_p_logits = self.latent_full.p_dist.logits + so_logits_p
        elif self.hyperparams['rubiz_params']['fusion_type'] == 'double_sigm':
            new_p_logits = torch.sigmoid(self.latent_full.p_dist.logits) * torch.sigmoid(so_logits_p)
        elif self.hyperparams['rubiz_params']['fusion_type'] == 'sigm':
            new_p_logits = self.latent_full.p_dist.logits * torch.sigmoid(so_logits_p)
        else:
            raise ValueError(self.hyperparams['rubiz_params']['fusion_type'])
        p_dist = self.latent_fused.dist_from_h(new_p_logits, mode)


        if mode == ModeKeys.TRAIN:
            latent_proj_q =  self.node_modules.get(self.node_type + "/rubiz_fusion_q", None)
            so_logits_q = self.latent_state.q_dist.logits
            if latent_proj_q is not None:
                so_logits_q = latent_proj_q(so_logits_q)
            if self.hyperparams['rubiz_params']['fusion_type'] == 'sum_of_logits':
                new_q_logits = self.latent_full.q_dist.logits + so_logits_q
            elif self.hyperparams['rubiz_params']['fusion_type'] == 'double_sigm':
                new_q_logits = torch.sigmoid(self.latent_full.q_dist.logits) * torch.sigmoid(so_logits_q)
            elif self.hyperparams['rubiz_params']['fusion_type'] == 'sigm':
                new_q_logits = self.latent_full.q_dist.logits * torch.sigmoid(so_logits_q)
            else:
                raise ValueError(self.hyperparams['rubiz_params']['fusion_type'])
            q_dist = self.latent_fused.dist_from_h(new_q_logits, mode)
            return p_dist, q_dist
        else:
            return p_dist

    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        x, x_so, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)

        x_so = x_so.detach()


        z, kl = self.encoder(mode, x, y_e, "full") # HEDI: kl between q(z|x,y) and p(z|x)
        z_so, kl_so = self.encoder(mode, x_so, y_e, "state")
        self.latent_fused.p_dist, self.latent_fused.q_dist = self.fuse_latents(mode)

        kl_fused = self.latent_fused.kl_q_p(self.log_writer, '%s' % str(self.node_type), self.curr_iter)
        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'kl_fused'), kl_fused, self.curr_iter)


        log_p_y_xz_fused, log_p_y_xz_so, log_p_y_xz_full = self.decoder(mode, x, x_so, x_nr_t, y, y_r, n_s_t0, z, z_so,
                                                          labels,  # Loss is calculated on unstandardized label
                                                          prediction_horizon,
                                                          self.hyperparams['k'])




        log_p_y_xz_fused_mean = torch.mean(log_p_y_xz_fused, dim=0)  # [nbs]
        log_likelihood_fused = torch.mean(log_p_y_xz_fused_mean)
        mutual_inf_p_fused = mutual_inf_mc(self.latent_fused.p_dist)
        mutual_inf_q_fused = mutual_inf_mc(self.latent_fused.q_dist)

        log_p_y_xz_so_mean = torch.mean(log_p_y_xz_so, dim=0)  # [nbs]
        log_likelihood_so = torch.mean(log_p_y_xz_so_mean)
        mutual_inf_q_so = mutual_inf_mc(self.latent_state.q_dist)
        mutual_inf_p_so = mutual_inf_mc(self.latent_state.p_dist)
        # HEDI: the mutual information measures if z is really dependant on x

        mutual_inf_p_full = mutual_inf_mc(self.latent_full.p_dist)
        log_p_y_xz_full_mean = torch.mean(log_p_y_xz_full, dim=0)  # [nbs]
        log_likelihood_full = torch.mean(log_p_y_xz_full_mean)

        loss_w = self.hyperparams['rubiz_params']['loss_weights']

        loss = -loss_w['full_ll']*log_likelihood_full + loss_w['full_kl']*self.kl_weight*kl - loss_w['full_mi']*mutual_inf_p_full
        loss += -loss_w['so_ll']*log_likelihood_so + loss_w['so_kl']*self.kl_weight*kl_so - loss_w['so_mi']*mutual_inf_p_so
        loss += -loss_w['fused_ll']*log_likelihood_fused + loss_w['fused_kl']*self.kl_weight*kl_fused - loss_w['fused_mi']*mutual_inf_p_fused


        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_y_xz_so'),
                                          log_p_y_xz_so_mean,
                                          self.curr_iter)
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_y_xz_fused'),
                                          log_p_y_xz_fused_mean,
                                          self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q_fused'),
                                       mutual_inf_q_fused,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p_fused'),
                                       mutual_inf_p_fused,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q_so'),
                                       mutual_inf_q_so,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p_so'),
                                       mutual_inf_p_so,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'll_full'),
                                       log_likelihood_full,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'll_fused'),
                                       log_likelihood_fused,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'll_so'),
                                       log_likelihood_so,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                       loss,
                                       self.curr_iter)
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(self.node_type), self.curr_iter)
        return loss

    def eval_loss(self,
                  inputs,
                  inputs_st,
                  first_history_indices,
                  labels,
                  labels_st,
                  neighbors,
                  neighbors_edge_value,
                  robot,
                  map,
                  prediction_horizon) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL

        x, x_so, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                               inputs=inputs,
                                                                               inputs_st=inputs_st,
                                                                               labels=labels,
                                                                               labels_st=labels_st,
                                                                               first_history_indices=first_history_indices,
                                                                               neighbors=neighbors,
                                                                               neighbors_edge_value=neighbors_edge_value,
                                                                               robot=robot,
                                                                               map=map)

        # DO THINGS HERE

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e, branch="full")  # [k_eval, nbs, N*K]
        z = self.latent_full.sample_p(1, mode, full_dist=True)
        y_dist, _ = self.p_y_xz(ModeKeys.PREDICT, x, x_nr_t, y_r, n_s_t0, z,
                                prediction_horizon, branch="full",
                                num_samples=1, num_components=num_components)
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)
        nll = -log_likelihood

        return nll

    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                num_samples,
                branch="full",
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                return_mode_prob=False):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, x_so, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                   inputs=inputs,
                                                                   inputs_st=inputs_st,
                                                                   labels=None,
                                                                   labels_st=None,
                                                                   first_history_indices=first_history_indices,
                                                                   neighbors=neighbors,
                                                                   neighbors_edge_value=neighbors_edge_value,
                                                                   robot=robot,
                                                                   map=map)


        self.latent_full.p_dist = self.p_z_x(mode, x, branch="full")
        self.latent_state.p_dist = self.p_z_x(mode, x_so, branch="state")

        if branch == "full":
            latent = self.latent_full
            X = x
        elif branch == "state":
            latent = self.latent_state
            X = x_so
        else:
            raise ValueError(branch)

        z, num_samples, num_components = latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        _, our_sampled_future = self.p_y_xz(mode, X, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            branch,
                                            num_components,
                                            gmm_mode)


        mode_prob = latent.p_dist.probs

        if return_mode_prob:
            return mode_prob, our_sampled_future
        else:
            return our_sampled_future

    def bias_full_kl(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, x_so, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                   inputs=inputs,
                                                                   inputs_st=inputs_st,
                                                                   labels=None,
                                                                   labels_st=None,
                                                                   first_history_indices=first_history_indices,
                                                                   neighbors=neighbors,
                                                                   neighbors_edge_value=neighbors_edge_value,
                                                                   robot=robot,
                                                                   map=map)


        self.latent_full.p_dist = self.p_z_x(mode, x, branch="full")
        self.latent_state.p_dist = self.p_z_x(mode, x_so, branch="state")


        z_full, num_samples, num_components = self.latent_full.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        z_state, num_samples, num_components = self.latent_state.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)


        y_dist_full, y_sample_full = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z_full,
                                            prediction_horizon,
                                            num_samples,
                                            "full",
                                            num_components,
                                            gmm_mode=False)

        y_dist_state, y_sample_state = self.p_y_xz(mode, x_so, x_nr_t, y_r, n_s_t0, z_state,
                                            prediction_horizon,
                                            num_samples,
                                            "state",
                                            num_components,
                                            gmm_mode=False)

        #KL(state, full) = Sum_{state}(log(pstate) - log(pfull))
        kl_sf = y_dist_state.log_prob(y_sample_state) - y_dist_full.log_prob(y_sample_state)
        kl_fs = y_dist_full.log_prob(y_sample_full) - y_dist_state.log_prob(y_sample_full)
        return kl_sf.mean(0, keepdims=True), kl_fs.mean(0, keepdims=True)