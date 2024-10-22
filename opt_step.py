from aMC_optimizer import aMC
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from nets import fcnn, Mushroomnet

torch.autograd.set_grad_enabled(False)


#this is where the neural-network parameters will be stored, and where you'll need to write the score
root = "run0/"
os.makedirs(root, exist_ok=True)

n = len([f for f in os.listdir(root) if f.startswith('batch')])
print('n is '+str(n))

#Bounds for the parameters
#these should stay fixed within the same optimization run

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


def net_to_protocol(n):
    params = torch.load(root+"batch"+str(n)+"/params.pt")
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


    np.savetxt(root+"batch"+str(n)+"/protocol.txt", temps)
    plt.plot(temps)
    plt.show()




if __name__ == "__main__":

    scale=.02

    net = Mushroomnet(1)

    init_net(net, scale)

    #optimize both the initial temperature and protocol
    opt =aMC([init_temp]+list(net.parameters()), init_sigma = scale, root = root)

    if n == 0:
        opt.sample_params()
        net_to_protocol(n) #transform sampled parameters to a protocol
        torch.save(opt.state_dict(), root+"batch"+str(n)+"/opt.pt")

    else:
        opt.load_state_dict(torch.load(root+"batch"+str(n-1)+"/opt.pt"))

        best_loss = opt.step()

        f = open(root+"losses.txt", "a")

        f.write(str(best_loss)+"\n")
        f.close()

        opt.sample_params()
        net_to_protocol(n)
        torch.save(opt.state_dict(), root+"batch"+str(n)+"/opt.pt")
