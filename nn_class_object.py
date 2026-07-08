
# Defining Layers and Activation Functions

class Layer:
    # constructor
    # x : input to layer
    # out : post-activation output
    def __init__(self):
        self.targ=None
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
        self.dw = self.x.T @ dq / self.targ.size
        self.db = np.sum(dq, axis=0, keepdims=True) / self.targ.size
        dz = dq @ self.w.T
        return(dz)

    def update_params(self, learn_rate:float) -> None:
        self.w -= learn_rate * self.dw 
        self.b -= learn_rate * self.db
    
    def get_params(self, learn_rate:float) -> None:
        return(self.w, self.b)

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
        self.y = None
        self.x = None
        self.loss = None 

    def __call__(self, x: np.ndarray, y:np.ndarray) -> float:
        return(self.fwd_loss(y, x))
    
    def fwd_loss(self, x:np.ndarray, y:np.ndarray) -> float:
        raise(NotImplementedError)
    
    def d_loss(self) -> np.ndarray:
        raise(NotImplementedError)

class MSE(Loss):
    # y: reconstruction
	# x: input, target
    def fwd_loss(self, x:np.ndarray, y:np.ndarray) -> float:
        self.y = y
        self.x = x
        self.loss = np.mean((y - x)**2)
        return(self.loss)

    def d_loss(self) -> np.ndarray:
        return(2 * (self.y - self.x) / (self.x.size))

# Multi-Layer Perceptron class
# Specifies forward loss abd backpropagation
# Updates Parameters
# Training iteration function

class MLP:
    # layers:list[Layer] # not inheriting from Layer; holding a compositional list of Layer objects
    # loss_fun: Loss # hold one loss-type object, no specificity
    def __init__(self, layers:list[Layer], loss_fun: Loss, learn_rate: float) -> None:
        self.layers = layers
        self.loss_fun = loss_fun
        self.learn_rate = learn_rate

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return(self.forward_pass(x))
    
    def forward_pass(self, x:np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward_pass(x)
        return(x)

    def loss(self, x:np.ndarray, y: np.ndarray) -> float:
        return(self.loss_fun(y, x))
    
    def backprop(self) -> None:
        dz = self.loss_fun.d_loss()
        for layer in reversed(self.layers):
            dz = layer.backprop(dz)

    def update_params(self) -> None:
        for layer in self.layers:
            layer.update_params(self.learn_rate)
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, cost_min: float) -> np.ndarray:
        recon = self.forward_pass(x)
        cost = self.loss(recon, x)
        cost_list = []
        epoch = 1

        while cost > cost_min:
            epoch += 1
            recon = self.forward_pass(x)
            cost = self.loss(recon, x)
            self.backprop()
            self.update_params()

            cost_list.append(cost)
        
        return(cost, cost_list, epoch)

    def save_params(self):
        w_list = []
        b_list = []
        n = len(self.layers)
        for layer in self.layers:
            w_list.append(layer.w)
            b_list.append(layer.b)
        
        np.savez('trained_model.npz', w_list, b_list)
        
        


'''
        losses = np.empty(epochs)
        for epoch in (pbar := trange(epochs)):
            running_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                pred = self.forward_pass(x_batch)
                running_loss += self.loss(pred, y_batch) * batch_size 
                self.backprop()
                self.update_params()

            running_loss /= len(x_train)
            pbar.set_description(f"Loss: {running_loss:.3f}")
            losses[epoch] = running_loss
        return(losses)
'''