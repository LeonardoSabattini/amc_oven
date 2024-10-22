from nets import fcnn, Mushroomnet
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy

torch.autograd.set_grad_enabled(False)
#number of times the temperature is updated within the experiment
num_time_steps = 100

#min and max temperature change for each update
min_rate = -10
max_rate = 10

#min and max temperature values
min_temp = 1150
max_temp = 1300

#initial temperature (in scaled units)
init_temp  = torch.tensor([0.5])



#all parameters are scaled between 0 and 1
#this turns them back to physical values
def unscale(values, min, max):
    return  values*(max-min) + min

def init_net(net, scale):
    for i, mod in enumerate(net.modules()):
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight, std=scale)
            if mod.bias is not None:
                torch.nn.init.normal_(mod.bias, std=scale)


def net_to_protocol(params):

    init_temp = torch.clip(params[0], 0.,1.)
    net_params = params[1:]
    for i,p in enumerate(net.parameters()):
        p.copy_(net_params[i])

    steps = torch.from_numpy(np.linspace(0,1, num_time_steps)).to(torch.float32)
    temps = np.zeros(num_time_steps)
    temps[0] = unscale(init_temp, min_temp, max_temp)
    for i in range(1,num_time_steps):
        out = net(steps[i].unsqueeze(0))
        rate = unscale(out, min_rate, max_rate)
        temps[i] = temps[i-1] + rate

        if temps[i] > max_temp:
            temps[i] = max_temp
        if temps[i] < min_temp:
            temps[i] = min_temp


    return temps

def plot_mutations(initial_params, num_mutations):
    def mutate_params(params):
        params = copy.deepcopy(params)
        for param in params:
            param.add_(scale*torch.randn(param.size()))
        return params

    protocols = []
    protocols.append(net_to_protocol(initial_params))
    for i in range(num_mutations):
        mut_params = mutate_params(initial_params)
        protocols.append(net_to_protocol(mut_params))
    
    for i in range(num_mutations):
        plt.plot(np.linspace(0,1,num_time_steps) + i, protocols[i])
    plt.axvline(1, ls = "--", color = "k")
    plt.ylim([min_temp, max_temp])
    plt.show()
    
scale=.02
net = Mushroomnet(1)
init_net(net, scale)


plot_mutations([init_temp]+list(net.parameters()),10)

