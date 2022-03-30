"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.

torch实现DDPG算法
"""
import torch
import numpy as np
import torch.nn as nn

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC_for_regression
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK

from torch.utils.data import DataLoader

from pytsk.gradient_descent.utils import NumpyDataLoader


seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)



# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, a_low, a_high):
        super(Actor,self).__init__()
        self.k = torch.tensor((a_high-a_low)/2.)
        self.b = torch.tensor((a_high+a_low)/2.)
        # layer
        self.layer_1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer_1.weight, 0., 0.3)
        nn.init.constant_(self.layer_1.bias, 0.1)
        # self.layer_1.weight.data.normal_(0.,0.3)
        # self.layer_1.bias.data.fill_(0.1)
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0.,0.3)
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        a = torch.relu(self.layer_1(s))
        a = torch.tanh(self.output(a))
        # 对action进行放缩，实际上a in [-1,1]
        scaled_a = a * self.k+self.b
        scaled_a = scaled_a.to(torch.float32)
        return scaled_a

# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic,self).__init__()
        n_layer = 30
        # layer
        self.layer_1 = nn.Linear(state_dim, n_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)
        
        self.layer_2 = nn.Linear(action_dim, n_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)
        
        self.output = nn.Linear(n_layer, 1)

    def forward(self,s,a):
        
        s = self.layer_1(s)
        a = self.layer_2(a)
        q_val = self.output(torch.relu(s+a))
        return q_val


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, a_low, a_high, s_low,s_high, replacement,memory_capacticy=1000,gamma=0.9,lr_a=0.001, lr_c=0.002,batch_size=32, train = True) :
        super(DDPG,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacticy = memory_capacticy
        self.s_low = s_low
        self.s_high = s_high
        self.a_low = a_low
        self.a_high = a_high
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        # 记忆库
        self.memory = np.zeros((memory_capacticy, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # 定义 Actor 网络
        # self.actor = Actor(state_dim, action_dim, a_low, a_high)
        # self.actor_target = Actor(state_dim, action_dim, a_low, a_high)
        n_rule = 10
        init_center = np.zeros((state_dim,n_rule))
        for i in range(state_dim):
            init_center[i] = np.arange(s_low[i]+(s_high[i]-s_low[i])/2/n_rule,s_high[i],(s_high[i]-s_low[i])/n_rule)
        gmf = nn.Sequential(
            AntecedentGMF(in_dim=state_dim, n_rule=n_rule, high_dim=True, init_center=init_center),
            nn.Dropout(p=0.2),
            nn.LayerNorm(n_rule),
            nn.ReLU()
        )
        self.actor = TSK(in_dim= state_dim, out_dim=action_dim, n_rule=n_rule, antecedent=gmf, order=1, precons=None)
        self.actor_target = TSK(in_dim= state_dim, out_dim=action_dim, n_rule=n_rule, antecedent=gmf, order=1, precons=None)
        if train:
            self.actor.train()
        else:
            self.actor.load_state_dict(torch.load("actor_models_constant_default_r/actor_episode_760.pkl"))
            self.actor_target.load_state_dict(torch.load("actor_models_lr0.001_rule10_DR0.2/actor_episode_1000.pkl"))
            self.actor.eval()
        
        # 定义 Critic 网络
        self.critic = Critic(state_dim,action_dim)
        self.critic_target = Critic(state_dim,action_dim)
        #self.load("critic.pkl")
        # 定义优化器
        ante_param, other_param = [], []
        for n, p in self.actor.named_parameters():
            if "center" in n or "sigma" in n:
                ante_param.append(p)
            else:
                other_param.append(p)
        weight_decay = 1e-8
        optimizer = torch.optim.AdamW(
            [{'params': ante_param, "weight_decay": 0},
            {'params': other_param, "weight_decay": weight_decay},],
            lr=lr_a
        )
        self.aopt = optimizer
        # self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()
        

    def sample(self):
        indices = np.random.choice(self.memory_capacticy, size=self.batch_size)
        return self.memory[indices, :] 

#    def choose_action(self, X):
#        """
#
#        Get the prediction of the model.
#
#        :param numpy.array X: Feature matrix :math:`X` with the size of :math:`[N, D]`.
#        :param y: Not used.
#        :return: Prediction matrix :math:`\hat{Y}` with the size of :math:`[N, C]`,
#            :math:`C` is the output dimension of the :code:`model`.
#
#        """
#        X = X.astype("float32")
#        test_loader = DataLoader(
#            NumpyDataLoader(X),
#            batch_size=self.batch_size,
#            shuffle=False,
#            num_workers=0,
#            drop_last=False
#        )
#        y_preds = []
#        for inputs in test_loader:
#            self.actor.eval()
#            y_pred = self.actor(inputs.to("cpu")).detach().cpu().numpy()
#            y_preds.append(y_pred)
#        return np.concatenate(y_preds, axis=0)
    
    
    def choose_action(self, s):
        s.astype("float32")
        s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()

    def learn(self):

        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            # for al in a_layers:
            #     al[1].weight.data.mul_((1-tau))
            #     al[1].weight.data.add_(tau * self.actor.state_dict()[al[0]+'.weight'])
            #     al[1].bias.data.mul_((1-tau))
            #     al[1].bias.data.add_(tau * self.actor.state_dict()[al[0]+'.bias'])
            for cl in c_layers:
                cl[1].weight.data.mul_((1-tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0]+'.weight'])
                cl[1].bias.data.mul_((1-tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0]+'.bias'])
                
            for n, p in self.actor_target.named_parameters():
                self.actor_target.state_dict()[n].mul_((1-tau))
                self.actor_target.state_dict()[n].add_(tau * self.actor.state_dict()[n])           
            
        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0]+'.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0]+'.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0]+'.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0]+'.bias']
            
            self.t_replace_counter += 1

        # 从记忆库中采样bacth data
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim])
        bs_ = torch.FloatTensor(bm[:,-self.state_dim:])
        
        # 训练Actor
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward()
        self.aopt.step()
        
        #训练critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target,q_eval)
        #print("{},{}".format(a_loss.detach().numpy(),td_error.detach().numpy()))
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s,a,[r],s_))
        index = self.pointer % self.memory_capacticy
        self.memory[index, :] = transition
        self.pointer += 1
        
    def save_critic(self ,path):
        torch.save(self.critic.state_dict(), path)
        
    def save_actor(self ,path):
        torch.save(self.actor.state_dict(), path)
    
    def load(self, path):
        """

        Load model.

        :param str path: Model save path.
        """
        self.critic.load_state_dict(torch.load(path))


    