import numpy as np
# Defining Layers and Activation Functions

class Layer:
    # constructor
    # x : input to layer
    # out : post-activation output
    def __init__(self):
        #self.targ=None
        self.x=None
        self.out=None
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        return self.forward_pass(x)

    # Ensure forward func is present
    def forward_pass(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    # Ensure backprop is present
    def backprop(self, dq: np.ndarray) -> np.ndarray:
        raise NotImplementedError 

    # Update params (weights and biases)
    def update_params(self, learn_rate: float) -> None:
        pass
    
    def get_params(self) -> None:
        pass
    

class Linear(Layer):
    # i : input dimension index
    # j : output dimension index (i + 1)
    def __init__(self, i: int, j: int):
        super().__init__()

        # Weights
        stdev = np.sqrt(2 / i)
        self.w = np.random.normal(0,stdev,size=(i,j))
        # same size as w
        self.dw = np.zeros_like(self.w)

        # Biases
        self.b = np.zeros((1, j))
        # same size as b
        self.db = np.zeros_like(self.b)
    
    def forward_pass(self, x:np.array) -> np.ndarray:
        self.x = x
        self.out = x @ self.w + self.b 
        return(self.out)

    def backprop(self, dq: np.ndarray) -> np.ndarray:
        # len(self.x) is the number of pixels in one channel
        self.dw = self.x.T @ dq / len(self.x)
        self.db = np.sum(dq, axis=0, keepdims=True) / len(self.x)
        dz = dq @ self.w.T
        return(dz)

    def update_params(self, learn_rate:float) -> None:
        self.w -= learn_rate * self.dw 
        self.b -= learn_rate * self.db
    
    def get_params(self) -> None:
        return(self.w, self.b)


# class Lin_act(Layer):
#     def forward_pass(self, x:np.ndarray) -> np.ndarray:
#         self.x = x
#         self.out = x
#         return(self.out)
    
#     def backprop(self, dq:np.ndarray) -> np.ndarray:
#         dz = dq
#         return(dz)
    
class ReLU(Layer):
    def forward_pass(self, x:np.ndarray) -> np.ndarray:
        self.x = x
        self.out = np.maximum(0,x)
        #self.out = x > 0
        return(self.out)
    
    def backprop(self, dq:np.ndarray) -> np.ndarray:
        dz = dq * (self.x > 0)
        return(dz)

class Tanh(Layer):
    def forward_pass(self, x:np.ndarray) -> np.ndarray:
        self.x = x
        self.out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return(self.out)
    
    def backprop(self, dq:np.ndarray) -> np.ndarray:
        dz = dq * (1 - self.out**2)
        return(dz)

# Defining Loss

class Loss:
    def __init__(self):
        self.recon = None
        self.img = None
        self.loss = None 

    def __call__(self, img: np.ndarray, recon:np.ndarray) -> float:
        return(self.fwd_loss(img, recon))
    
    def fwd_loss(self, img:np.ndarray, recon:np.ndarray) -> float:
        raise(NotImplementedError)
    
    def d_loss(self) -> np.ndarray:
        raise(NotImplementedError)

class MSE(Loss):
    # y: reconstruction
	# x: input, target
    def fwd_loss(self, img:np.ndarray, recon:np.ndarray) -> float:
        self.recon = recon
        self.img = img
        self.loss = np.mean((recon - img)**2)
        return(self.loss)

    def d_loss(self) -> np.ndarray:
        return(2 * (self.recon - self.img) / len(self.img)) #(self.img.size))

# Multi-Layer Perceptron class
# Specifies forward loss and backpropagation
# Updates Parameters
# Training iteration function

class MLP:
    # layers:list[Layer] # not inheriting from Layer; holding a compositional list of Layer objects
    # loss_fun: Loss # hold one loss-type object, no specificity
    def __init__(self, layers:list[Layer], bneck_ind:int, loss_fun: Loss, learn_rate: float) -> None:
        self.layers = layers
        self.arch_len = len(layers)
        self.loss_fun = loss_fun
        self.learn_rate = learn_rate
        self. bneck_ind =  bneck_ind

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return(self.forward_pass(x))
    
    def forward_pass(self, x:np.ndarray) -> np.ndarray:
        # x: input to layer
        for ind, layer in enumerate(self.layers):
            x = layer.forward_pass(x)
            if ind == self.bneck_ind:
                bottleneck = x
        return(x, bottleneck)

    def loss(self, img:np.ndarray, recon: np.ndarray) -> float:
        # img: input data
        # recon: reconstruction
        return(self.loss_fun(img, recon))
    
    def backprop(self) -> None:
        dz = self.loss_fun.d_loss()
        for layer in reversed(self.layers):
            dz = layer.backprop(dz)

    def update_params(self) -> None:
        for layer in self.layers:
            layer.update_params(self.learn_rate)
    
    def opt_lr(self, img: np.ndarray, epochs:int) -> np.ndarray:
        cost_list = []

        for j in range(epochs):

            recon, bottleneck = self.forward_pass(img)
            cost = self.loss(img, recon)
            cost_list.append(cost)

            self.backprop()
            self.update_params()

            print("Learning rate:", self.learn_rate)
            print("Epoch:", j)
            print("Cost:", cost)
        
       #print(len(cost_list))
        return(np.asarray(cost_list))


    def train(self, img: np.ndarray, cost_min: float) -> np.ndarray:
        # img: input data
        epoch = 1
        recon, bottleneck = self.forward_pass(img)
        cost = self.loss(img, recon)
        cost_list = [cost]

        while cost > cost_min:
            print("Epoch number:", epoch)
            print("Cost:", cost)
            self.backprop()
            self.update_params()

            recon, bottleneck = self.forward_pass(img)
            cost = self.loss(img, recon)

            cost_list.append(cost)
            epoch += 1
        
        return(cost_list, recon, bottleneck)

    #def save_params(self, bottleneck, recon, epoch, cost_list):
    def save_params(self):
        w_dict = {}
        b_dict = {}
        n = 0

        for layer in self.layers:
            weights_biases = layer.get_params()
            if weights_biases != None:
                w,b = weights_biases
                w_dict[f"w{n}"] = w
                b_dict[f"b{n}"] = b
                n += 1

        np.savez('trained_model.npz', **w_dict, **b_dict) #, bottleneck=bottleneck, output=recon)

