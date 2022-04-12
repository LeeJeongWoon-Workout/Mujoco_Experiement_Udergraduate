import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import matplotlib.pyplot as plt
import copy
from mixed_replay_buffer_stochastic import mixed_replay_buffer_stochastic
from vanilla_episodic_buffer import vanilla_episodic_buffer
from tac import TAC
from sac import SAC
from d2rl_tac import D2RL_TAC
from d2rl_sac import D2RL_SAC
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: True)')
parser.add_argument('--epoch',type=int,default=100000,metavar='G')
parser.add_argument('--render',type=bool,default=False,metavar='G')
parser.add_argument('--iteration',type=int,default=10,metavar='G')



args = parser.parse_args()
# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)
result_list=list()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

test_env=list()
for i in range(args.iteration):
    test=gym.make(args.env_name)
    test.seed(args.seed+i)
    test_env.append(test)

agent1=D2RL_TAC(env.observation_space.shape[0], env.action_space, args=args, q=1.2)
agent2=D2RL_TAC(env.observation_space.shape[0], env.action_space, args=args, q=1.4)
agent3=D2RL_TAC(env.observation_space.shape[0], env.action_space, args=args, q=1.5)
agent4=D2RL_SAC(env.observation_space.shape[0], env.action_space, args=args)
agent5=TAC(env.observation_space.shape[0], env.action_space, args=args, q=1.5)
agent6=TAC(env.observation_space.shape[0], env.action_space, args=args, q=1.2)
agent7=SAC(env.observation_space.shape[0], env.action_space, args=args)
agent_list=[agent1,agent2,agent3,agent4,agent5,agent6,agent7]

mixed_memory1=vanilla_episodic_buffer(args.replay_size,args.seed,tau=0.0)
mixed_memory2=vanilla_episodic_buffer(args.replay_size,args.seed,tau=0.0)
mixed_memory3=vanilla_episodic_buffer(args.replay_size,args.seed,tau=0.0)
mixed_memory4=vanilla_episodic_buffer(args.replay_size,args.seed,tau=0.0)
mixed_memory5=vanilla_episodic_buffer(args.replay_size,args.seed,tau=0.0)
mixed_memory6=vanilla_episodic_buffer(args.replay_size,args.seed,tau=0.0)
mixed_memory7=vanilla_episodic_buffer(args.replay_size,args.seed,tau=0.0)

memory_list=[mixed_memory1,mixed_memory2,mixed_memory3,mixed_memory4,mixed_memory5,mixed_memory6,mixed_memory7]


num_list1=list()
num_list2=list()
num_list3=list()
num_list4=list()
num_list5=list()
num_list6=list()
num_list7=list()
num_list=[num_list1,num_list2,num_list3,num_list4,num_list5,num_list6,num_list7]

reward_list1=list()
reward_list2=list()
reward_list3=list()
reward_list4=list()
reward_list5=list()
reward_list6=list()
reward_list7=list()
reward_list=[reward_list1,reward_list2,reward_list3,reward_list4,reward_list5,reward_list6,reward_list7]

total_numsteps1=0
total_numsteps2=0
total_numsteps3=0
total_numsteps4=0
total_numsteps5=0
total_numsteps6=0
total_numsteps7=0
total_numstep_list=[total_numsteps1,total_numsteps2,total_numsteps3,total_numsteps4,total_numsteps5,total_numsteps6,total_numsteps7]

updates=0

R_MAX1=-1000000000000000000000000000
R_MAX2=-1000000000000000000000000000
R_MAX3=-1000000000000000000000000000
R_MAX4=-1000000000000000000000000000
R_MAX5=-1000000000000000000000000000
R_MAX6=-1000000000000000000000000000
R_MAX7=-1000000000000000000000000000
R_MAX_list=[R_MAX1,R_MAX2,R_MAX3,R_MAX4,R_MAX5,R_MAX6,R_MAX7]

for i in range(7): # 알고리
    agent_i=agent_list.pop(0)
    memory_i=memory_list.pop(0)
    R_MAX_i=R_MAX_list.pop(0)
    total_num_i=total_numstep_list.pop(0)
    #i : index of algorithm
    for iteration in range(1): # 훈련 반복
        # iteration : experiment
        agent = agent_i
        memory=memory_i
        R_MAX=R_MAX_i
        total_num=total_num_i
        for i_episode in itertools.count(1):
            #i_episode: episode each experiment
            episode_reward = 0
            episode_steps = 0
            done = False
            state = env.reset()
            transition_list = list()
            while not done:
                if args.render:
                    env.render()
                if args.start_steps>total_num:
                    action=env.action_space.sample()
                else:
                    action=agent.select_action(state)

                if len(memory)>args.batch_size:

                    for s in range(args.updates_per_step):

                        agent.update_parameters(memory,args.batch_size,updates)

                        updates+=1

                next_state, reward, done, _ = env.step(action)


                mask = 1 if episode_steps == env._max_episode_steps else float(not done)

                memory.push(state, action, reward, next_state, mask)  # Append transition to memory
                transition_list.append([state, action, reward, next_state, mask])
                total_num+=1
                state=next_state
                episode_steps += 1
                episode_reward += reward

            if episode_reward>R_MAX:
                R_MAX=episode_reward
                for t in transition_list:
                    memory.hpush(*t)

            if i_episode%5==0:
                for iter in range(args.iteration):# test 반복

                    episode_reward = 0
                    episode_steps = 0
                    done=False
                    state=test_env[iter].reset()
                    print("total_num_step: {} , algorithm: {}  Evaluating: {}".format(total_num,i,iter))
                    while not done:
                        action=agent.select_action(state,evaluate=True)
                        next_state,reward,done,_=test_env[iter].step(action)
                        state=next_state
                        episode_steps+=1
                        episode_reward+=reward

                    reward_list[i].append(episode_reward)
                    num_list[i].append(total_num)
            if total_num>args.num_steps:
                break
#for csv
name_list=["D2RL_TAC q=1.2","D2RL_TAC q=1.4","D2RL_TAC q=1.5","D2RL_SAC","TAC q=1.5","TAC q=1.2","SAC"]
csv_algo_list=list()
csv_inter_list=list()
csv_reward_list=list()

for i in range(7):
    name=[name_list[i] for n in range(len(num_list[i]))]
    csv_algo_list.extend(name)
    csv_inter_list.extend(num_list[i])
    csv_reward_list.extend(reward_list[i])



my_df=pd.DataFrame({"Model":csv_algo_list,"Interaction":csv_inter_list,"Accumulated Reward":csv_reward_list})
plt.figure(figsize=(30,10))
plt.title(args.env_name)
sns.lineplot(x="Interaction",y="Accumulated Reward",hue="Model",data=my_df)
plt.savefig('{}.png'.format(args.env_name))



