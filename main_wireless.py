import argparse
import copy
from collections import defaultdict
from copy import deepcopy
import cv2
from PIL import Image, ImageDraw
import importlib
import math
import os
import random
from distutils.util import strtobool

import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import supersuit as ss
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import utils
from utils import gather_actions_2d, gather_next_actions_2d, gather_critic_score, get_all_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="attn", help="model architecture")
    parser.add_argument("--gym-id", type=str, default="wireless_comm",
                        help="the id of the gym environment")
    parser.add_argument("--grid-size", type=int, default=5, help="grid size of the environment")
    parser.add_argument("--pkg-p", type=float, default=0.5, help="package arrival probability")
    parser.add_argument("--success-p", type=float, default=0.8, help="package transmit success probability")
    parser.add_argument("--ddl", type=int, default=2, help="observed deadline horizon")

    parser.add_argument("--max-cycles", type=int, default=12, help="maximum number of redeploy steps")
    parser.add_argument("--test-every", type=int, default=20, help="evaluation frequency")
    parser.add_argument("--actor-lr", type=float, default=5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--critic-lr", type=float, default=5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--cpu", action='store_true',
                        help="if toggled, this experiment will be run on CPU only")
    parser.add_argument("--total-timesteps", type=int, default=800,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", action='store_true',
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--debug", action='store_true',
                        help="if toggled, this experiment will be visualized")
    parser.add_argument("--anneal-lr", action='store_true',
                        help="if toggled, we perform learning rate annealing")

    # Algorithm specific arguments
    parser.add_argument("--n-sample-traj", type=int, default=15,
                        help="number of trajectories to sample for occupancy measure & policy gradient")
    parser.add_argument("--n-sample-Q-steps", type=int, default=256,
                        help="number of trajectories to sample for Q evaluations")
    parser.add_argument("--target-network-frequency", type=int, default=1,
                        help="how often to update the target Q networks")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="how much to update the target Q networks each time")
    parser.add_argument("--gamma", type=float, default=0.7,
                        help="the discount factor gamma")
    parser.add_argument("--reward-scale", type=float, default=1,
                        help="divide reward by reward scale")
    parser.add_argument("--update-epochs", type=int, default=1,
                        help="the K epochs to update the Q networks")
    parser.add_argument("--max-grad-norm", type=float, default=1,
                        help="the maximum norm for the gradient clipping")

    # CDSAC specific arguments
    parser.add_argument("--n-obs-neighbors", type=int, default=1, help="neighborhood size (kappa)")
    parser.add_argument("--eta-mu", type=float, default=0, help="constraint weight")
    parser.add_argument("--rhs", type=float, default=2, help="rhs of constraint")

    args = parser.parse_args()
    return args


class Agent(nn.Module):
    def __init__(self, num_states, num_actions, num_ddl):
        super().__init__()
        self.state_layer1 = self._layer_init(nn.Linear(num_states, 64))
        self.state_layer2 = self._layer_init(nn.Linear(64 * num_ddl, 256))
        self.state_layer3 = self._layer_init(nn.Linear(256, 64))
        self.actor = self._layer_init(nn.Linear(64, num_actions), std=0.01)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer, 'bias'):
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_action(self, x, action=None, return_prob=False):
        state_latent = torch.flatten(self.state_layer1(x), start_dim=1)
        logits = self.actor(F.relu(self.state_layer3(F.relu(self.state_layer2(F.relu(state_latent))))))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        if return_prob:
            return action, probs.log_prob(action), probs.entropy(), probs.probs
        else:
            return action, probs.log_prob(action), probs.entropy()


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, n_obs_neighbors, num_ddl):
        super().__init__()
        self.state_layer1 = self._layer_init(nn.Linear(num_states, 64))
        self.action_layer1 = self._layer_init(nn.Embedding(num_actions + 1, 8))
        self.critic = self._layer_init(nn.Linear(64 * num_ddl + 8 * ((2 * n_obs_neighbors + 1) ** 2 - 1), 256))
        self.critic1 = self._layer_init(nn.Linear(256, 64))
        self.critic2 = self._layer_init(nn.Linear(64, num_actions), std=1.0)
        self.num_actions = num_actions
        self.n_obs_neighbors = n_obs_neighbors

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer, 'bias'):
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x, action):
        state_latent = torch.flatten(self.state_layer1(x), start_dim=1)
        action_latent = torch.flatten(self.action_layer1(action), start_dim=1)
        return self.critic2(F.relu(self.critic1(F.relu(self.critic(F.relu(torch.cat([state_latent, action_latent],
                                                                                    dim=-1)))))))


def find_neighbors(grid_x, grid_y, n_obs_neighbors):
    neighbor_dict = defaultdict(list)
    n_neighbor_dict = defaultdict(int)
    for i in range(grid_x):
        for j in range(grid_y):
            agent_id = i * grid_y + j
            for l in range(i - n_obs_neighbors, i + n_obs_neighbors + 1):
                for m in range(j - n_obs_neighbors, j + n_obs_neighbors + 1):
                    if l < 0 or m < 0 or l >= grid_x or m >= grid_y:
                        neighbor_dict[agent_id].append(-1)
                    else:
                        neighbor_dict[agent_id].append(l * grid_y + m)
                        n_neighbor_dict[agent_id] += 1
            neighbor_dict[agent_id].remove(agent_id)
    return neighbor_dict, n_neighbor_dict


def image_background(grid_x, grid_y, agent_shape, agent_gap):
    small_gap = (agent_gap - agent_shape) // 2
    w = (grid_y + 2) * agent_shape + (grid_y - 1) * agent_gap
    h = (grid_x + 2) * agent_shape + (grid_x - 1) * agent_gap
    im = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for i in range(grid_x):
        for j in range(grid_y):
            draw.rectangle([agent_shape + j * (agent_shape + agent_gap),
                            agent_shape + i * (agent_shape + agent_gap),
                            agent_shape * 2 + j * (agent_shape + agent_gap),
                            agent_shape * 2 + i * (agent_shape + agent_gap)],
                           fill=(141, 219, 252),
                           outline='black')
    for i in range(grid_x - 1):
        for j in range(grid_y - 1):
            leftUpPoint = (agent_shape + agent_shape * (j + 1) + j * agent_gap + small_gap,
                           agent_shape + agent_shape * (i + 1) + i * agent_gap + small_gap)
            rightDownPoint = (leftUpPoint[0] + agent_shape,
                              leftUpPoint[1] + agent_shape)
            draw.ellipse([leftUpPoint, rightDownPoint], fill=(167, 252, 141), outline="black")
    return np.array(im)


if __name__ == "__main__":
    args = parse_args()
    save_every_step = 50
    test_every_step = args.test_every
    gamma = args.gamma
    max_cycles = args.max_cycles
    num_updates = args.total_timesteps
    warmup = num_updates // 3 * 2

    init_n_sample_traj = args.n_sample_traj
    n_sample_traj = init_n_sample_traj
    n_obs_neighbors = args.n_obs_neighbors
    grid_x = args.grid_size
    grid_y = args.grid_size
    num_agents = grid_x * grid_y
    neighbor_dict, n_neighbor_dict = find_neighbors(grid_x, grid_y, n_obs_neighbors)
    agent_shape = 40
    agent_gap = 80
    visual_background = image_background(grid_x, grid_y, agent_shape, agent_gap)
    num_ddl = args.ddl
    pkg_p, success_p = args.pkg_p, args.success_p
    dual_mu = 0.1
    eta_mu = args.eta_mu
    entropy_constraint = args.rhs
    reward_scale = args.reward_scale
    n_sample_Q_steps = args.n_sample_Q_steps
    run_name = f"{args.gym_id}_{num_ddl}x{grid_x}x{grid_y}_P{pkg_p}xQ{success_p}_{num_agents}AG_N{n_obs_neighbors}" \
               f"_Eta{eta_mu}_rhs{entropy_constraint}_seed{args.seed}_gm{args.gamma}_{utils.name_with_datetime()}"
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)

    # TRY NOT TO MODIFY: seeding
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_file = importlib.import_module(f'envs.{args.gym_id}')
    # env setup
    env = env_file.parallel_env(
        grid_x=grid_x, grid_y=grid_y, ddl=num_ddl, packet_arrival_probability=pkg_p,
        success_transmission_probability=success_p, n_obs_neighbors=n_obs_neighbors, max_iter=max_cycles
    )

    num_agents = len(env.possible_agents)
    # agents 0...num_agents are ordered from left to right
    num_actions = env.action_space(env.possible_agents[0]).n
    obs_size = (n_obs_neighbors * 2 + 1) ** 2
    num_states = obs_size

    if args.track:
        wandb_tags = []
        wandb.init(entity="ANONYMOUS",
                   project="ANONYMOUS",
                   name=run_name,
                   sync_tensorboard=True,
                   monitor_gym=True, config={
                'model': __file__[:__file__.find('.py')],
                'actor_lr': args.actor_lr,
                'critic_lr': args.critic_lr,
                'anneal_lr': args.anneal_lr,
                'gamma': args.gamma,
                'eta_mu': eta_mu,
                'max_cycles': max_cycles,
                'reward_scale': reward_scale,
                'entropy_constraint': entropy_constraint,
                'n_sample_traj': n_sample_traj,
                'n_sample_Q_steps': n_sample_Q_steps,
                'n_obs_neighbors': n_obs_neighbors,
                'update_epochs': args.update_epochs,
                'max_grad_norm': args.max_grad_norm},
                   # notes="",
                   tags=wandb_tags,
                   save_code=True
                   )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None

    actors = [Agent(num_states, num_actions, num_ddl) for i in range(num_agents)]
    critics1 = [Critic(num_states, num_actions, n_obs_neighbors, num_ddl) for i in range(num_agents)]
    critics1_target = deepcopy(critics1)
    critics1_copy = [Critic(num_states, num_actions, n_obs_neighbors, num_ddl) for i in range(num_agents)]
    critics1_copy_target = deepcopy(critics1_copy)
    critics2 = [Critic(num_states, num_actions, n_obs_neighbors, num_ddl) for i in range(num_agents)]
    critics2_target = deepcopy(critics2)
    critics2_copy = [Critic(num_states, num_actions, n_obs_neighbors, num_ddl) for i in range(num_agents)]
    critics2_copy_target = deepcopy(critics2_copy)
    actor_optimizer = [optim.Adam(x.parameters(), lr=args.actor_lr) for x in actors]
    critic1_optimizer = [optim.Adam(x.parameters(), lr=args.critic_lr) for x in critics1]
    critic1_copy_optimizer = [optim.Adam(x.parameters(), lr=args.critic_lr) for x in critics1_copy]
    critic2_optimizer = [optim.Adam(x.parameters(), lr=args.critic_lr) for x in critics2]
    critic2_copy_optimizer = [optim.Adam(x.parameters(), lr=args.critic_lr) for x in critics2_copy]

    for x in actors + critics1 + critics1_copy + critics2 + critics2_copy:
        x.to(device)
    for x in critics1_target + critics1_copy_target + critics2_target + critics2_copy_target:
        x.to(device)
        for p in x.parameters():
            p.requires_grad = False

    # ALGO Logic: Storage setup
    rb_q_obs = torch.zeros(n_sample_Q_steps, num_agents, num_ddl, num_states).to(device)
    rb_q_obs_next = torch.zeros(n_sample_Q_steps, num_agents, num_ddl, num_states).to(device)
    rb_q_actions = torch.zeros(n_sample_Q_steps, num_agents, dtype=torch.int64).to(device)
    rb_q_rewards1 = torch.zeros(n_sample_Q_steps, num_agents).to(device)
    rb_q_rewards2 = torch.zeros(n_sample_Q_steps, num_agents).to(device)
    rb_q_terms = torch.zeros(n_sample_Q_steps, num_agents).to(device)

    global_step = 0
    pbar = trange(1, num_updates + 1)
    for update in pbar:
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            ac_lrnow = frac * args.actor_lr
            cr_lrnow = frac * args.critic_lr
            for opt_temp in actor_optimizer:
                opt_temp.param_groups[0]["lr"] = ac_lrnow
            for opt_temp in critic1_optimizer + critic1_copy_optimizer + critic2_optimizer + critic2_copy_optimizer:
                opt_temp.param_groups[0]["lr"] = cr_lrnow

        warmup_ratio = max(0, (warmup - update) / max(1, warmup))
        total_episodic_return = torch.zeros(n_sample_traj, num_agents).to(device)

        rb_obs = torch.zeros(n_sample_traj, max_cycles, num_agents, num_ddl, num_states).to(device)
        rb_actions = torch.zeros(n_sample_traj, max_cycles, num_agents, dtype=torch.int64).to(device)
        rb_terms = torch.zeros(n_sample_traj, max_cycles, num_agents).to(device)
        rb_occupancy_measure = torch.zeros(n_sample_traj, num_agents, num_actions).to(device)
        rb_end_steps = torch.zeros(n_sample_traj, dtype=torch.long).to(device)

        # Step 3-5: collect n_sample_traj trajectories and find average occupancy measure
        for traj_id in range(n_sample_traj):
            next_obs = env.reset()

            for step in range(0, max_cycles):
                global_step += 1

                obs = utils.batchify(next_obs, device)

                with torch.no_grad():
                    # get action from the agent
                    actions = torch.cat([actors[i].get_action(obs[i].unsqueeze(0))[0] for i in range(num_agents)],
                                        dim=0)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    utils.unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[traj_id, step] = obs
                rb_actions[traj_id, step] = actions
                rb_terms[traj_id, step] = utils.batchify(terms, device)

                # compute episodic return
                total_episodic_return[traj_id] += utils.batchify(rewards, device)
                rb_end_steps[traj_id] = step

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    break

            # bootstrap value if not done
            with torch.no_grad():
                rb_occupancy_measure[traj_id] = F.one_hot(rb_actions[traj_id, rb_end_steps[traj_id]],
                                                          num_classes=num_actions)
                for t in reversed(range(rb_end_steps[traj_id])):
                    rb_occupancy_measure[traj_id] = F.one_hot(rb_actions[traj_id, t], num_classes=num_actions) \
                                                    + gamma * rb_occupancy_measure[traj_id]

        avg_occupancy_measure = torch.mean(rb_occupancy_measure, dim=0)
        avg_occupancy_measure[:, 0] = 0

        # flatten the batch
        keep_mask = torch.arange(max_cycles).unsqueeze(0).repeat(n_sample_traj, 1).to(device)
        keep_mask = keep_mask <= rb_end_steps.unsqueeze(-1)
        b_obs = rb_obs[keep_mask]
        b_actions = rb_actions[keep_mask]
        b_terms = rb_terms[keep_mask]

        avg_episodic_return = total_episodic_return.mean().item()
        avg_end_step = rb_end_steps.to(torch.float32).mean().item()
        n_sample_traj = math.floor(init_n_sample_traj * max_cycles / avg_end_step)
        if args.track:
            writer.add_scalar("episode_details/episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("episode_details/end_step", avg_end_step, global_step)
            writer.add_scalar("episode_details/n_sample_traj", n_sample_traj, global_step)
        pbar.set_description(f'Total return: {avg_episodic_return:.2f}')


        # Step 6: collect n_sample_Q_steps to update the value functions
        def r_mu(pos_y):  # entropy >= c
            select_occupancy = avg_occupancy_measure[torch.arange(pos_y.shape[0], device=device), pos_y]
            return ((1 - gamma) ** 2) * select_occupancy


        # rollout for data collection
        with torch.no_grad():
            reset = True
            for q_step in range(n_sample_Q_steps):
                if reset:
                    next_obs = env.reset()
                    reset = False
                obs = utils.batchify(next_obs, device)

                # get action from the agent
                actions = torch.cat([actors[i].get_action(obs[i].unsqueeze(0))[0] for i in range(num_agents)], dim=0)

                next_obs, rewards, terms, _, infos = env.step(
                    utils.unbatchify(actions, env)
                )

                if len(next_obs) == 0:
                    reset = True
                    q_step -= 1
                else:
                    rb_q_obs[q_step] = obs
                    rb_q_obs_next[q_step] = utils.batchify(next_obs, device)
                    rb_q_actions[q_step] = actions
                    rb_q_rewards1[q_step] = utils.batchify(rewards, device) / reward_scale
                    rb_q_rewards2[q_step] = r_mu(actions)
                    rb_q_terms[q_step] = utils.batchify(terms, device)

        if args.track:
            writer.add_scalar("episode_details/avg_reward1", rb_q_rewards1.mean().item(), global_step)
            writer.add_scalar("episode_details/avg_reward2", rb_q_rewards2.mean().item(), global_step)

        # optimize the value networks
        """
        if args.debug:
            print('=' * 20)
            vis_ind = torch.randint(0, n_sample_Q_steps, (3,), device=device)
        """
        for agent_id in range(num_agents):
            critic1_loss_track = []
            critic2_loss_track = []

            # get critic_values
            neighbor_actions = gather_actions_2d(rb_q_actions, neighbor_dict, n_obs_neighbors, num_actions, agent_id,
                                                 device)
            neighbor_next_actions, agent_next_actions = gather_next_actions_2d(rb_q_obs_next, actors, neighbor_dict,
                                                                               n_obs_neighbors, agent_id,
                                                                               num_actions, device)
            critic1_scores = get_all_scores(critics1, critics1_copy, critics1_target, critics1_copy_target,
                                            rb_q_obs[:, agent_id], rb_q_obs_next[:, agent_id],
                                            rb_q_actions[:, agent_id], agent_next_actions,
                                            agent_id, neighbor_actions, neighbor_next_actions)
            critic1_current_score, critic1_copy_current_score, critic1_next_score, \
                critic1_copy_next_score = critic1_scores

            critic1_final_score = torch.minimum(critic1_next_score, critic1_copy_next_score)
            critic1_target_score = (1 - rb_q_terms[:, agent_id]) * gamma * critic1_final_score + \
                                   rb_q_rewards1[:, agent_id]

            loss_critic1 = F.mse_loss(critic1_current_score, critic1_target_score.detach())
            critic1_optimizer[agent_id].zero_grad(set_to_none=True)
            loss_critic1.backward()
            nn.utils.clip_grad_norm_(critics1[agent_id].parameters(), args.max_grad_norm)
            critic1_optimizer[agent_id].step()
            loss_critic1_copy = F.mse_loss(critic1_copy_current_score, critic1_target_score.detach())
            critic1_copy_optimizer[agent_id].zero_grad(set_to_none=True)
            loss_critic1_copy.backward()
            nn.utils.clip_grad_norm_(critics1_copy[agent_id].parameters(), args.max_grad_norm)
            critic1_copy_optimizer[agent_id].step()
            critic1_loss_track.append(loss_critic1.item())
            """
            if args.debug:
                print(f'AG {agent_id} reward: {rb_q_rewards1[vis_ind, agent_id].data.cpu().numpy()}, '
                      f'critic1: {critic1_current_score[vis_ind].data.cpu().numpy()}, '
                      f'copy: {critic1_copy_current_score[vis_ind].data.cpu().numpy()}, '
                      f'target: {critic1_target_score[vis_ind].data.cpu().numpy()}')
            """

            critic2_scores = get_all_scores(critics2, critics2_copy, critics2_target, critics2_copy_target,
                                            rb_q_obs[:, agent_id], rb_q_obs_next[:, agent_id],
                                            rb_q_actions[:, agent_id], agent_next_actions,
                                            agent_id, neighbor_actions, neighbor_next_actions)
            critic2_current_score, critic2_copy_current_score, critic2_next_score, \
                critic2_copy_next_score = critic2_scores
            critic2_final_score = torch.minimum(critic2_next_score, critic2_copy_next_score)
            critic2_target_score = (1 - rb_q_terms[:, agent_id]) * gamma * critic2_final_score + \
                                   rb_q_rewards2[:, agent_id]

            loss_critic2 = F.mse_loss(critic2_current_score, critic2_target_score.detach())
            critic2_optimizer[agent_id].zero_grad()
            loss_critic2.backward()
            nn.utils.clip_grad_norm_(critics2[agent_id].parameters(), args.max_grad_norm)
            critic2_optimizer[agent_id].step()
            loss_critic2_copy = F.mse_loss(critic2_copy_current_score, critic2_target_score.detach())
            critic2_copy_optimizer[agent_id].zero_grad()
            loss_critic2_copy.backward()
            nn.utils.clip_grad_norm_(critics2_copy[agent_id].parameters(), args.max_grad_norm)
            critic2_copy_optimizer[agent_id].step()
            critic2_loss_track.append(loss_critic2.item())
            """
            if args.debug:
                print(f'AG {agent_id} reward: {rb_q_rewards2[vis_ind, agent_id].data.cpu().numpy()}, '
                      f'critic2: {critic2_current_score[vis_ind].data.cpu().numpy()}, '
                      f'copy: {critic2_copy_current_score[vis_ind].data.cpu().numpy()}, '
                      f'target: {critic2_target_score[vis_ind].data.cpu().numpy()}')
            """

            # update the target networks
            for param, target_param in zip(critics1[agent_id].parameters(),
                                           critics1_target[agent_id].parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(critics1_copy[agent_id].parameters(),
                                           critics1_copy_target[agent_id].parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(critics2[agent_id].parameters(),
                                           critics2_target[agent_id].parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(critics2_copy[agent_id].parameters(),
                                           critics2_copy_target[agent_id].parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if args.track and agent_id % 5 == 0:
                writer.add_scalar(f"losses/agent_{agent_id}_critic1_loss",
                                  sum(critic1_loss_track) / len(critic1_loss_track), global_step)
                writer.add_scalar(f"losses/agent_{agent_id}_critic2_loss",
                                  sum(critic2_loss_track) / len(critic2_loss_track), global_step)

        # step 7: Lagrangian multiplier update
        occupancy_entropy = 0.5 * ((1 - gamma) ** 2) * torch.sum(
            avg_occupancy_measure * avg_occupancy_measure, dim=1)
        total_violation = torch.sum(torch.clamp(-occupancy_entropy + entropy_constraint, min=0))
        dual_mu = torch.clamp(-eta_mu * (occupancy_entropy - entropy_constraint), min=0)

        if args.track:
            writer.add_scalar(f"episode_details/dual_mu", torch.mean(dual_mu), global_step)
            writer.add_scalar(f"episode_details/occupancy_entropy", torch.mean(occupancy_entropy), global_step)
            writer.add_scalar(f"episode_details/total_violation", total_violation, global_step)

        # step 8-9: policy gradient evaluation & parameter update
        policy_bsz = b_obs.shape[0]
        for agent_id in range(num_agents):
            with torch.no_grad():
                critic1_score_tracker = []
                critic2_score_tracker = []
                actor_neighbor_actions = gather_actions_2d(b_actions, neighbor_dict, n_obs_neighbors, num_actions,
                                                           agent_id, device)

                # agent_obs, agent_action
                critic1_neighbor_score = gather_critic_score(critics1, b_obs[:, agent_id],
                                                             b_actions[:, agent_id], agent_id,
                                                             actor_neighbor_actions)
                critic1_copy_neighbor_score = gather_critic_score(critics1_copy, b_obs[:, agent_id],
                                                                  b_actions[:, agent_id], agent_id,
                                                                  actor_neighbor_actions)
                critic2_neighbor_score = gather_critic_score(critics2, b_obs[:, agent_id],
                                                             b_actions[:, agent_id], agent_id,
                                                             actor_neighbor_actions)
                critic2_copy_neighbor_score = gather_critic_score(critics2_copy, b_obs[:, agent_id],
                                                                  b_actions[:, agent_id], agent_id,
                                                                  actor_neighbor_actions)

                critic1_final_score = torch.minimum(critic1_neighbor_score, critic1_copy_neighbor_score)
                critic2_final_score = torch.minimum(critic2_neighbor_score, critic2_copy_neighbor_score)

                neighbor_critic_score1 = critic1_final_score + dual_mu[agent_id] * critic2_final_score
                neighbor_score_sum = neighbor_critic_score1.clone().detach()
                critic1_score_tracker.append(torch.mean(critic1_neighbor_score).item())
                critic2_score_tracker.append(torch.mean(critic2_neighbor_score).item())

                for neighbor_id in neighbor_dict[agent_id]:
                    neighbor_actions = gather_actions_2d(b_actions, neighbor_dict, n_obs_neighbors,
                                                         num_actions, neighbor_id, device)

                    critic1_neighbor_score = gather_critic_score(critics1, b_obs[:, neighbor_id],
                                                                 b_actions[:, neighbor_id], neighbor_id,
                                                                 neighbor_actions)
                    critic1_copy_neighbor_score = gather_critic_score(critics1_copy, b_obs[:, neighbor_id],
                                                                      b_actions[:, neighbor_id], neighbor_id,
                                                                      neighbor_actions)
                    critic2_neighbor_score = gather_critic_score(critics2, b_obs[:, neighbor_id],
                                                                 b_actions[:, neighbor_id], neighbor_id,
                                                                 neighbor_actions)
                    critic2_copy_neighbor_score = gather_critic_score(critics2_copy, b_obs[:, neighbor_id],
                                                                      b_actions[:, neighbor_id], neighbor_id,
                                                                      neighbor_actions)

                    critic1_final_score = torch.minimum(critic1_neighbor_score, critic1_copy_neighbor_score)
                    critic2_final_score = torch.minimum(critic2_neighbor_score, critic2_copy_neighbor_score)

                    neighbor_critic_score2 = critic1_final_score + dual_mu[neighbor_id] * critic2_final_score
                    neighbor_score_sum = neighbor_score_sum + neighbor_critic_score2
                    critic1_score_tracker.append(torch.mean(critic1_neighbor_score).item())
                    critic2_score_tracker.append(torch.mean(critic2_neighbor_score).item())

                average_score = neighbor_score_sum / n_neighbor_dict[agent_id]
                average_score = (1 - warmup_ratio) * average_score + warmup_ratio * neighbor_critic_score1
                average_critic1_score = sum(critic1_score_tracker) / len(critic1_score_tracker)
                average_critic2_score = sum(critic2_score_tracker) / len(critic2_score_tracker)

            _, agent_logprob, _ = actors[agent_id].get_action(b_obs[:, agent_id], action=b_actions[:, agent_id])
            pg_loss = torch.mean(-average_score.detach() * agent_logprob)
            actor_optimizer[agent_id].zero_grad(set_to_none=True)
            pg_loss.backward()
            nn.utils.clip_grad_norm_(actors[agent_id].parameters(), args.max_grad_norm)
            actor_optimizer[agent_id].step()

            if args.track and agent_id % 5 == 0:
                writer.add_scalar(f"losses/agent_{agent_id}_policy_loss", pg_loss.item(), global_step)
                writer.add_scalar(f"losses/agent_{agent_id}_critic1_score", average_critic1_score, global_step)
                writer.add_scalar(f"losses/agent_{agent_id}_critic2_score", average_critic2_score, global_step)

            """
            if (update + 1) % save_every_step == 0:
                utils.save_checkpoint({'global_step': global_step,
                                       'state_dict': agent.state_dict(),
                                       'optim_dict': optim.state_dict()},
                                      global_step=global_step,
                                      checkpoint=f"runs/{run_name}")
            """

        if args.debug and (update + 1) % test_every_step == 0:
            with torch.no_grad():
                actions = torch.zeros(num_agents, dtype=torch.int64, device=device)
                action_probs = torch.zeros(num_agents, num_actions, device=device)
                critic1_score = torch.zeros(num_agents, device=device)
                critic2_score = torch.zeros(num_agents, device=device)
                total_test_episode_return = torch.zeros(num_agents, device=device)
                for sample_id in range(2):
                    frames = [Image.fromarray(np.zeros_like(visual_background))]
                    next_obs = env.reset()

                    for step in range(0, max_cycles):
                        obs = utils.batchify(next_obs, device)

                        # get action from the agent
                        for agent_id in range(num_agents):
                            agent_action, _, _, agent_probs = actors[agent_id].get_action(
                                obs[agent_id].unsqueeze(0),
                                return_prob=True)
                            actions[agent_id] = agent_action
                            action_probs[agent_id] = agent_probs

                        action_probs_np = action_probs.data.cpu().numpy()

                        # execute the environment and log data
                        next_obs, rewards, terms, truncs, infos = env.step(
                            utils.unbatchify(actions, env)
                        )

                        reward_list = list(rewards.values())
                        obs_np = obs.data.cpu().numpy()
                        img = copy.deepcopy(visual_background)
                        for agent_id in range(num_agents):
                            text_start_x = agent_shape * (1 + agent_id % grid_y) + agent_gap * (agent_id % grid_y)
                            text_start_y = agent_shape * (2 + agent_id // grid_y) + agent_gap * (agent_id // grid_y)
                            cv2.putText(img,
                                        f"{obs_np[agent_id, :, n_obs_neighbors * (2 * n_obs_neighbors + 1) + n_obs_neighbors].tolist()}",
                                        (text_start_x, text_start_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
                            if actions[agent_id] == 0:
                                continue
                            start_x = agent_shape * (1.5 + agent_id % grid_y) + agent_gap * (agent_id % grid_y)
                            start_y = agent_shape * (1.5 + agent_id // grid_y) + agent_gap * (agent_id // grid_y)
                            arrow_color = reward_list[agent_id] == 1
                            if actions[agent_id] == 1:
                                arrow_start_x, arrow_start_y = start_x - agent_shape // 2, start_y - agent_shape // 2
                                arrow_end_x, arrow_end_y = arrow_start_x - agent_gap // 2, \
                                                           arrow_start_y - agent_gap // 2
                            elif actions[agent_id] == 2:
                                arrow_start_x, arrow_start_y = start_x - agent_shape // 2, start_y + agent_shape // 2
                                arrow_end_x, arrow_end_y = arrow_start_x - agent_gap // 2, \
                                                           arrow_start_y + agent_gap // 2
                            elif actions[agent_id] == 3:
                                arrow_start_x, arrow_start_y = start_x + agent_shape // 2, start_y - agent_shape // 2
                                arrow_end_x, arrow_end_y = arrow_start_x + agent_gap // 2, \
                                                           arrow_start_y - agent_gap // 2
                            elif actions[agent_id] == 4:
                                arrow_start_x, arrow_start_y = start_x + agent_shape // 2, start_y + agent_shape // 2
                                arrow_end_x, arrow_end_y = arrow_start_x + agent_gap // 2, \
                                                           arrow_start_y + agent_gap // 2
                            img = cv2.arrowedLine(img, (int(arrow_start_x), int(arrow_start_y)),
                                                  (int(arrow_end_x), int(arrow_end_y)),
                                                  (246, 123, 210) if arrow_color else (0, 0, 0), 2)

                        frames.append(Image.fromarray(img))

                        # compute episodic return
                        total_test_episode_return += utils.batchify(rewards, device)

                        # if we reach termination or truncation, end
                        if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                            break

                    frames[0].save(f"./runs/{run_name}/step_{update}_sample_{sample_id}_"
                                   f"return_{total_test_episode_return.mean().item():.3f}.gif", save_all=True,
                                   append_images=frames[1:], optimize=False, duration=2000, loop=1)

    env.close()
    if args.track:
        writer.close()