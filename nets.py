import torch

class netBlock(torch.nn.Module):
    def __init__(self, n_in, n_out, layernorm = True, residual = True):
        super().__init__()
        if residual and n_in != n_out:
            print(error)
        
        if layernorm:
            self.block = torch.nn.Sequential(torch.nn.Linear(n_in, n_out), torch.nn.LayerNorm(n_out, elementwise_affine=False), torch.nn.Tanh())
        else:
            self.block = torch.nn.Sequential(torch.nn.Linear(n_in, n_out),  torch.nn.Tanh())
        self.residual = residual

    def forward(self,x):
        if self.residual:
            return x + self.block(x)
        else:
            return self.block(x)

class fcnn(torch.nn.Module):
    def __init__(self,n_in, n_out, num_layers, hidden_nodes):
        super().__init__()

        model = []

        model += [netBlock(n_in, hidden_nodes, residual = False)]

        for i in range(num_layers):
            model += [netBlock(hidden_nodes, hidden_nodes)]

        model += [torch.nn.Linear(hidden_nodes,n_out), torch.nn.Sigmoid()]         

        self.model = torch.nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)


class rnn(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden):
        super().__init__()

        # self.rnn_cell = torch.nn.GRUCell(n_in, n_hidden)
        self.rnn_cell = torch.nn.RNNCell(n_in, n_hidden)
        self.linear = torch.nn.Linear(n_hidden, n_out)
        # self.out = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_out), torch.nn.Tanh())

    def forward(self, x, hidden):
        hidden = self.rnn_cell(x,hidden)
        out = torch.tanh(self.linear(hidden))
        return out,hidden
    
class Mushroomnet(torch.nn.Module):
    def __init__(self,n_in):
        super().__init__()
        hidden_dim=4
        act = torch.nn.Tanh()
        model = []
        
        model += [torch.nn.Linear(n_in, hidden_dim), torch.nn.LayerNorm(hidden_dim, elementwise_affine = False), act]

        for i in range(4):
            model += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim, elementwise_affine = False), act]
        
        model += [torch.nn.Linear(hidden_dim, 10), torch.nn.LayerNorm(10, elementwise_affine = False), act]
        model += [torch.nn.Linear(10, 1, bias=False), torch.nn.Sigmoid()]
        self.model = torch.nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)
    