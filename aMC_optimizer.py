
import torch
import numpy as np
import copy
import os
import json



class aMC(torch.optim.Optimizer):
    def __init__(self, params,init_sigma,epsilon = 1e-2, n_reset = 20, sigma_decay = .95, root = "./"):
        

        defaults = dict(init_sigma = init_sigma, epsilon = epsilon, n_reset = n_reset ,sigma_decay = sigma_decay)
        super().__init__(params,defaults)
        self.root = root
        #initialize optimizer parameters
        
        # NOTE: aMC currently has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self.param_groups[0]['params'][0]]
        

        state['best_loss'] = -np.inf
        state['n_reset'] = n_reset
        state['sigma'] = init_sigma
        state['n_cr'] = 0
        state['mus'] = []
        state['epsilon'] = epsilon
        state['best_params'] = []
        state['sigma_decay'] = sigma_decay
        for group in self.param_groups:
            for p in group['params']:
                state['mus'].append(torch.zeros_like(p))


        for group in self.param_groups:
            for p in group['params']:
                state['best_params'].append(p)

        state['epoch'] = 0
        
    def sample_params(self):
        state = self.state[self.param_groups[0]['params'][0]]
        os.makedirs(self.root+"/batch"+str(state['epoch']), exist_ok=True)

        params = copy.deepcopy(state['best_params'])
        noise = copy.deepcopy(state['best_params'])

        for group in self.param_groups:
            for i in range(len(group['params'])):
                noise[i] = torch.normal(mean =  state['mus'][i], std= state['sigma'])

                params[i] += noise[i]

        torch.save(params, self.root+"batch"+str(state['epoch'])+"/params.pt")
        torch.save(noise, self.root+"batch"+str(state['epoch'])+"/noise.pt")

    @torch.no_grad()
    def step(self):

        state = self.state[self.param_groups[0]['params'][0]]

        #read losses
        with open(self.root + "/batch"+str(state['epoch'])+'/results.json') as f:
            loss = json.load(f)
        print(loss)


        if loss > state['best_loss']:

            state['n_cr'] = 0
            state['best_loss'] = loss
            new_params = torch.load( self.root+"/batch"+str(state['epoch'])+"/params.pt")
            for i,p in enumerate(new_params):
                state['best_params'][i] = p
            noise = torch.load( self.root+"batch"+str(state['epoch'])+"/noise.pt")
            if state['epoch'] > 0:
                for group in self.param_groups:
                    for i in range(len(group['params'])):
                        state['mus'][i] += state['epsilon'] * (noise[i] - state['mus'][i])

        else:
            state['n_cr'] +=1

            if state['n_cr'] == state['n_reset']:
                state['sigma'] *= state['sigma_decay']
                for group in self.param_groups:
                    for i in range(len(group['params'])):
                        state['mus'][i].zero_()

                state['n_cr'] = 0
                        

        state['epoch'] += 1
        return state['best_loss']
