import copy
import math

from torch.utils.hipify.hipify_python import bcolors

from funcs.model_SR import *
import torch.optim as optim
from funcs.parameters import *
import torch
import numpy as np
import math


class Ppo:
    def __init__(self, N_S, N_A, ac_net, cr_net, regulation_method, alpha, epsilon, batch_size, lr, delta, usePo=False,
                 regnn2=0.0, regnn1=0.0,regL2_2=0.0,regL2_1=0.0):
        self.actor_net = Actor(N_S, N_A, ac_net)  # 两层网络
        self.critic_net = Critic(N_S, cr_net)  # 两层网络
        # self.cri_orig = Criticorig(N_S, cr_net)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=lr)
        # self.critic_optim = optim.Adam(self.critic_net.parameters(),lr=lr_critic,weight_decay=l2_rate)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=lr)
        # self.critic_loss_func = torch.nn.MSELoss()
        # loss define
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.usePo = usePo
        self.regulation_method = regulation_method
        self.alpha = alpha
        self.delta = delta
        self.regnn2=regnn2
        self.regnn1=regnn1
        self.regL2_2=regL2_2
        self.regL2_1=regL2_1
        if (regulation_method == 'Hoyer'):
            self.critic_loss_func = my_loss_hoyer_w(self.critic_net, alpha)  # 自定义的loss函数
        elif (regulation_method == 'Hoyeraw'):
            self.critic_loss_func = my_loss_hoyer_aw(self.critic_net, alpha,regnn2,regnn1,regL2_2,regL2_1)  # 自定义的loss函数
        elif (regulation_method == 'L1'):
            self.critic_loss_func = my_loss_L1_w(self.critic_net, alpha)  # 自定义的loss函数
        elif (regulation_method == 'Hoyeraw_re') or regulation_method == 'Hoyeraw_re1':
            self.critic_loss_func = my_loss_hoyer_aw_re(self.critic_net, alpha,regnn2,regnn1,regL2_2,regL2_1)  # 自定义的loss函数
        elif (regulation_method == 'L1_re'):
            self.critic_loss_func = my_loss_L1_w_re(self.critic_net, alpha)  # 自定义的loss函数
        elif (regulation_method == 'L2'):
            self.critic_loss_func = my_loss_L2_w(self.critic_net, alpha)  # 自定义的loss函数
        elif (regulation_method == 'log'):
            self.critic_loss_func = my_loss_log_w(self.critic_net, alpha, delta)  # 自定义的loss函数
        elif (regulation_method == 'None'):
            self.critic_loss_func = my_loss_L1_w(self.critic_net, 0)  # 自定义的loss函数

    def train(self, memory):
        printgrad=False
        anomalcheck=False
        memory = np.array(memory, dtype=object)
        states = torch.tensor(np.vstack(memory[:, 0]), dtype=torch.float32)
        actions = torch.tensor(np.vstack(memory[:, 1]), dtype=torch.float32)#actions = torch.tensor(list(memory[:, 1]), dtype=torch.float32)
        rewards = torch.tensor(list(memory[:, 2]), dtype=torch.float32)
        masks = torch.tensor(list(memory[:, 3]), dtype=torch.float32)
        # actions = torch.tensor(np.array(memory[:,1],'single'),dtype=torch.float32)
        # rewards = torch.tensor(np.array(memory[:,2],'single'),dtype=torch.float32)
        # masks = torch.tensor(np.array(memory[:,3],'single'),dtype=torch.float32)

        # values：critic网络对states的价值预测；
        values, rep_L2, rep_L1 = self.critic_net(states)

        returns, advants = self.get_gae(rewards, masks, values)
        old_mu, old_std = self.actor_net(states)
        pi = self.actor_net.distribution(old_mu, old_std)
        # [min(old_mu), max(old_mu), min(old_std), max(old_std)] [min(mu), max(mu), min(std), max(std)]

        old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)

        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n // self.batch_size):
                b_index = arr[self.batch_size * i:self.batch_size * (i + 1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                mu, std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu, std)
                new_prob = pi.log_prob(b_actions).sum(1, keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                # KL散度正则项
                # KL_penalty = self.kl_divergence(old_mu[b_index],old_std[b_index],mu,std)
                ratio = torch.exp(new_prob - old_prob)

                surrogate_loss = ratio * b_advants

                values, rep_L2, rep_L1 = self.critic_net(b_states)
                if (self.regulation_method == 'Hoyeraw' or self.regulation_method=='Hoyeraw_re' or self.regulation_method == 'Hoyeraw_re1'):
                    critic_loss = self.critic_loss_func(values, b_returns, rep_L2, rep_L1)  # critic 更新
                else:
                    critic_loss = self.critic_loss_func(values, b_returns)  # critic 更新

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                if self.usePo:
                    if self.regulation_method == 'L1' or self.regulation_method=='L1_re':
                        # 1-模po
                        self.norm1_po(self.critic_net, self.alpha)  # 1-模po
                    elif self.regulation_method == 'Hoyer' or self.regulation_method=='Hoyeraw' or self.regulation_method=='Hoyeraw_re1':
                        self.hoyer1_po(self.critic_net, self.alpha)
                        # self.hoyer1w_po(self.critic_net, self.alpha) #changed on 0623
                    elif self.regulation_method == 'Hoyeraw_re':
                        # self.hoyer1_po(self.critic_net, self.alpha)
                        self.hoyer1w_po(self.critic_net, self.alpha) #changed on 0623
                    elif self.regulation_method == 'log':
                        self.normlog_po(self.critic_net, self.alpha, self.delta)

                ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                clipped_loss = ratio * b_advants
                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()
                # actor_loss = -(surrogate_loss-beta*KL_penalty).mean()

                self.actor_net_old=copy.deepcopy(self.actor_net)
                self.actor_optim.zero_grad()  # actor 更新 torch.autograd.set_detect_anomaly(True)

                if not anomalcheck:
                    actor_loss.backward()
                else:
                    with torch.autograd.detect_anomaly():
                        actor_loss.backward()
                # # print grad check
                # if printgrad:
                #     v_n = []
                #     v_v = []
                #     v_g = []
                #     for name, parameter in self.actor_net.named_parameters():
                #         v_n.append(name)
                #         v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                #         v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
                #     for i in range(len(v_n)):
                #         if np.max(np.abs(v_g[i])).item()<1e-6:
                #         #if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
                #             color = bcolors.FAIL + '*'
                #             if not anomalcheck:
                #                 anomalcheck=True
                #         else:
                #             color = bcolors.OKGREEN + ' '
                #         print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                #         print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))

                self.actor_optim.step()
                if torch.isnan(self.actor_net.fc1.weight).any():
                    # model=self.actor_net
                    # for name, module in model.named_modules():
                    #     module.register_full_backward_hook(get_backward_hook(name))
                    self.actor_net.state_dict()['fc1.weight'].copy_(self.actor_net_old.state_dict()['fc1.weight'])
                    self.actor_net.state_dict()['fc1.bias'].copy_(self.actor_net_old.state_dict()['fc1.bias'])
                    self.actor_net.state_dict()['fc2.weight'].copy_(self.actor_net_old.state_dict()['fc2.weight'])
                    self.actor_net.state_dict()['fc2.bias'].copy_(self.actor_net_old.state_dict()['fc2.bias'])
                    self.actor_net.state_dict()['mu.weight'].copy_(self.actor_net_old.state_dict()['mu.weight'])
                    self.actor_net.state_dict()['mu.bias'].copy_(self.actor_net_old.state_dict()['mu.bias'])
                    self.actor_net.state_dict()['sigma.weight'].copy_(self.actor_net_old.state_dict()['sigma.weight'])
                    self.actor_net.state_dict()['sigma.bias'].copy_(self.actor_net_old.state_dict()['sigma.bias'])
                    # self.actor_net=self.actor_net_old

    #

    # 计算KL散度, 并没有使用,本意是用于PPO1
    def kl_divergence(self, old_mu, old_sigma, mu, sigma):

        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()
        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / \
             (2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    # 计算GAE
    def get_gae(self, rewards, masks, values):
        # values为critic网络预测值
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            # 计算A_t并进行加权求和
            # running_returns：带折扣累计奖励值
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            # running_tderror = r + gamma * v(st+1) - v(st)
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - \
                              values.data[t]
            # running_advants = running_tderror + gamma * lambd * running_advants;
            running_advants = running_tderror + gamma * lambd * \
                              running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        # advants的归一化
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    # 进行L1的近端下降
    def norm1_po(self, model, alpha):
        model_dict = model.state_dict()
        for name, param in model.named_parameters():
            if 'weight' in name:
                kernel = param.clone().detach()
                sign = torch.sign(kernel)
                temp = torch.abs(kernel) - alpha
                zeros_tensor = torch.zeros_like(temp)
                kernel1 = torch.where(temp < 0, zeros_tensor, temp) * sign
                model_dict[name] = kernel1
        model.load_state_dict(model_dict)

    def hoyer1_po(self, model, alpha):
        model_dict = model.state_dict()
        for name, param in model.named_parameters():
            if 'weight' in name:
                kernel = param.clone().detach()
                zeros_tensor = torch.zeros_like(kernel)
                zL1 = torch.sum(torch.abs(kernel))
                zL2 = torch.sqrt(torch.sum(torch.square(kernel))) + 1e-15
                t = alpha / zL2
                kernel1 = torch.where(torch.abs(kernel) < t, zeros_tensor, kernel)
                model_dict[name] = kernel1
        model.load_state_dict(model_dict)

    def hoyer1w_po(self, model, alpha):
        model_dict = model.state_dict()
        for name, param in model.named_parameters():
            if 'weight' in name:
                kernel = param.clone().detach()
                zeros_tensor = torch.zeros_like(kernel)
                zL1 = torch.sum(torch.abs(kernel))
                zL2 = torch.sqrt(torch.sum(torch.square(kernel))) + 1e-15
                c1 = alpha / (zL2 ** 3)
                t = alpha / zL2
                kernel1 = torch.where(torch.abs(kernel) < t,
                                      zeros_tensor, (kernel-np.sign(kernel)*t)/(1-c1*zL1))
                model_dict[name] = kernel1
        model.load_state_dict(model_dict)

    def normlog_po(self, model, alpha, delta):
        model_dict = model.state_dict()
        for name, param in model.named_parameters():
            if 'weight' in name:
                kernel = param.clone().detach()
                bound = 2 * math.sqrt(alpha) - delta
                kernel_zeros = torch.zeros_like(kernel)
                temp = torch.abs(kernel)
                newresult = 0.5 * (
                            kernel - torch.sign(kernel) * (delta - torch.sqrt(torch.pow(temp + delta, 2) - 4 * alpha)))
                kernel1 = torch.where(temp >= bound, newresult, kernel_zeros)
                model_dict[name] = kernel1
        model.load_state_dict(model_dict)


class my_loss_hoyer_w(torch.nn.Module):
    def __init__(self, model, reg_nn):
        super().__init__()
        self.model = model
        self.reg_nn = reg_nn

    def forward(self, x, y):
        mse = torch.mean(torch.pow((x - y), 2))
        regularization_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # print(param)
                regularization_loss += torch.square(torch.sum(torch.abs(param))) / torch.sum(torch.square(param))
        return mse + self.reg_nn * regularization_loss


class my_loss_hoyer_aw(torch.nn.Module):
    def __init__(self, model, reg_nn,regnn2,regnn1,regL2_2,regL2_1):
        super().__init__()
        self.model = model
        self.reg_nn = reg_nn
        self.regnn2=regnn2
        self.regnn1=regnn1
        self.regL2_2=regL2_2
        self.regL2_1=regL2_1

    def forward(self, x, y, rep_L2, rep_L1):
        mse = torch.mean(torch.pow((x - y), 2))
        regularization_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # print(param)
                regularization_loss += torch.square(torch.sum(torch.abs(param))) / torch.sum(torch.square(param))
        rep_hoyer_loss = torch.square(torch.sum(torch.abs(rep_L2))) / torch.sum(torch.square(rep_L2)) *self.regnn2+\
                         torch.square(torch.sum(torch.abs(rep_L1))) / torch.sum(torch.square(rep_L1)) * self.regnn1
        rep_L2_loss=torch.sum(torch.square(rep_L2))*self.regL2_2+torch.sum(torch.square(rep_L1))*self.regL2_1
        return mse + self.reg_nn * regularization_loss+rep_hoyer_loss+rep_L2_loss

class my_loss_hoyer_aw_re(torch.nn.Module):
    def __init__(self, model, reg_nn,regnn2,regnn1,regL2_2,regL2_1):
        super().__init__()
        self.model = model
        self.reg_nn = reg_nn
        self.regnn2=regnn2
        self.regnn1=regnn1
        self.regL2_2=regL2_2
        self.regL2_1=regL2_1

    def forward(self, x, y, rep_L2, rep_L1):
        mse = torch.mean(torch.pow((x - y), 2))
        regularization_loss = 0
        for name, param in self.model.named_parameters():
            if 'fc3' not in name:
                if 'weight' in name:
                    # print(param)
                    regularization_loss += torch.square(torch.sum(torch.abs(param))) / (torch.sum(torch.square(param))+1e-15)
        rep_hoyer_loss = torch.square(torch.sum(torch.abs(rep_L2))) / (torch.sum(torch.square(rep_L2))+1e-15) *self.regnn2+\
                         torch.square(torch.sum(torch.abs(rep_L1))) / (torch.sum(torch.square(rep_L1))+1e-15) * self.regnn1
        rep_L2_loss=torch.sum(torch.square(rep_L2))*self.regL2_2+torch.sum(torch.square(rep_L1))*self.regL2_1
        return mse + self.reg_nn * regularization_loss+rep_hoyer_loss+rep_L2_loss

class my_loss_L1_w(torch.nn.Module):
    def __init__(self, model, reg_nn):
        super().__init__()
        self.model = model
        self.reg_nn = reg_nn

    def forward(self, x, y):
        mse = torch.mean(torch.pow((x - y), 2))
        regularization_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                regularization_loss += torch.sum(torch.abs(param))
        return mse + self.reg_nn * regularization_loss

class my_loss_L1_w_re(torch.nn.Module):
    def __init__(self, model, reg_nn):
        super().__init__()
        self.model = model
        self.reg_nn = reg_nn

    def forward(self, x, y):
        mse = torch.mean(torch.pow((x - y), 2))
        regularization_loss = 0
        for name, param in self.model.named_parameters():
            if 'fc3' not in name:
                if 'weight' in name:
                    regularization_loss += torch.sum(torch.abs(param))
        return mse + self.reg_nn * regularization_loss

class my_loss_L2_w(torch.nn.Module):
    def __init__(self, model, reg_nn):
        super().__init__()
        self.model = model
        self.reg_nn = reg_nn

    def forward(self, x, y):
        mse = torch.mean(torch.pow((x - y), 2))
        regularization_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                regularization_loss += torch.sum(torch.square(param))
        return mse + self.reg_nn * regularization_loss


class my_loss_log_w(torch.nn.Module):
    def __init__(self, model, reg_nn, delta):
        super().__init__()
        self.model = model
        self.reg_nn = reg_nn
        self.delta = delta

    def forward(self, x, y):
        mse = torch.mean(torch.pow((x - y), 2))
        regularization_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                regularization_loss += torch.sum(torch.log(1 + torch.abs(param) / self.delta))
        return mse + self.reg_nn * regularization_loss
