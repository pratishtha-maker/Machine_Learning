import numpy as np

def create_layers(depth, in_dim, hidden, out_dim):
    layers = []
    layers.append(get_layer(in_ch = in_dim, out_ch = hidden, act_func = 'sigmoid', weight_init='zeroes'))
    for i in range(depth-1):
        layers.append(get_layer(in_ch = hidden, out_ch = hidden, act_func = 'sigmoid', weight_init='zeroes'))
    layers.append(get_layer(in_ch=hidden, out_ch=out_dim, act_func = 'val', weight_init='zeroes', bias=False))
    return layers

class get_layer:
    def sigmoid(self, x):
        sigma = 1 / (1 + np.exp(-x))
        return sigma

    def sigmoid_bar(self, x):
        sigma = 1 / (1 + np.exp(-x))
        return sigma * (1 - sigma)

    def val(self, x):
        return x

    def val_bar(self, x):
        return 1

    def __init__(self, in_ch, out_ch, act_func, weight_init, bias = True):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.act_func = act_func

        if bias:
            shape = (self.in_ch+1, self.out_ch+1)
        else:
            shape = (self.in_ch+1, self.out_ch)

        if weight_init == 'zeroes':
            self.layer_weights = np.zeros(shape, dtype=np.float)
        elif weight_init == 'random':
            self.layer_weights = np.random.standard_normal(shape)
        else: raise NotImplementedError
            
    def __str__(self) -> str:
        return str(self.layer_weights)
    
    def evaluation(self, x):
        if self.act_func == 'sigmoid':
            return self.sigmoid(np.dot(x, self.layer_weights))
        else:
            return np.dot(x, self.layer_weights)
    
    def backprop(self, zs, partials):
        delta = np.dot(partials[-1], self.layer_weights.T)
        if self.act_func == "sigmoid":
            delta *= self.sigmoid_bar(zs)
            return delta
        else:
            return delta

    
    def update_ws(self, lr, zs, partials):
        grad = zs.T.dot(partials)
        self.layer_weights += -lr * grad
        return grad