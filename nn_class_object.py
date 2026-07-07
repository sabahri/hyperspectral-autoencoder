
##############################
###### Defining Layers #######
##############################

class Layer:
    # constructor
    # x : input to layer
    # out : post-activation output
    def __init__(self):
        self.x=None
        self.out=None
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        return self.forward(x)

    # Ensure forward func is present
    def forward(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    # Ensure backprop is present
    def backward(self, dq: np.ndarray) -> np.ndarray:
        raise NotImplementedError 

    # Update params (weights and biases)
    def update_params(self, learn_rate: float) -> None:
        pass

class Linear(Layer):
    # i : input dimension index
    # j : output dimension index (i + 1)
    def __init__(self, i: int, j: int):
        super().__init__()

        stdev = np.sqrt(2 / i)
        self.w = np.random.normal(0,stdev,size=(i,j))
        # same size as w
        self.dw = np.zeros_like(self.w)

        self.b = np.zeros((1, j))
        # same size as b
        self.db = np.zeros_like(self.b)
    
    def forward(self, x:np.array) -> np.ndarray:
        self.x = x
        self.out = np.dot(x, self.w) + self.b 
        return(self.out)

    def backward(self, dq: np.ndarray) -> np.ndarray:
        self.dw = np.dot(self.x.T, dq)
        self.db = np.sum(dq, axis=0, keepdims=True)
        dz = np.dot(dq, self.w.T)
        return(dz)

    def update_params(self, learn_rate:float) -> None:
        self.w -= learn_rate * self.dw 
        self.b -= learn_rate * self.db

class ReLU(Layer):
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.x = x
        self.out = np.max(0,x)
        #self.out = x > 0
        return(self.out)
    
    def backward(self, dq:np.ndarray) -> np.ndarray:
        dz = dq * (x > 0)
        return(dz)

class Tanh(Layer):
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.x = x
        self.out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return(self.out)
    
    def backward(self, dq:np.ndarray) -> np.ndarray:
        dz = dq * (1 - self.out**2)
        return(dz)

##############################
######## Defining Loss #######
##############################

class Loss:
    def __init__(self):
        self.y = None
        self.x = None
        self.loss = None 

    def __call__(self, y:np.ndarray, x: np.ndarray) -> float:
        return(self.forward(y, x))
    
    def forward(self, y np.ndarray, x:np.ndarray) -> float:
        raise(NotImplementedError)
    
    def backward(self) -> np.ndarray:
        raise(NotImplementedError)

class MSE(Loss):
    # y: reconstruction
	# x: target
    def mse_loss(self, x:np.ndarray, y:np.ndarray) -> float:
        self.y = y
        self.x = x
        self.loss = np.mean((y - x)**2)
        return(self.loss)

    def mse_der(self) -> np.ndarray:
        return(2 * (self.y - self.x) / (self.x.size))

# Multi-Layer Perceptron class
# Specifies forward loss abd backpropagation
# Updates Parameters
# Training iteration function

class MLP:
    def __init__(self, layers:list[Layer], loss_fun: Loss, learn_rate: float) -> None:
        self.layers = layers
        self.loss_fn = loss_fn
        self.learn_rate = learn_rate

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return(self.forward(x))
    
    def forward(self, x:np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return(x)

    def loss(self, x:np.ndarray, y: np.ndarray) -> float:
        return(self.loss_fn(y, x))
    
    def backward(self) -> None:
        dz = self.loss_fn.backward()
        for layer in reversed(self.layers):
            dz = layer.backward(dz)

    def update(self) -> None:
        for layer in self.layers:
            layer.step(self.learn_rate)
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int) -> np.ndarray:
        losses = np.empty(epochs)
        for epoch in (pbar := trange(epochs)):
            running_loss = 0.0
            for i in range(0, len(x_train), batch_size):