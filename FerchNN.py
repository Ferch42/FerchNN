import numpy as np
from scipy.special import expit

class NN_Inside_Layer:    
    
    def __init__(self, input_size, number_of_neurons = 30 , activation = "sigmoid"):
        
        self.input_size = input_size
        self.number_neurons = number_of_neurons
        self.activation = activation
        self.weights = self.glorot_uniform((input_size, number_of_neurons))
        self.biases = self.glorot_uniform(number_of_neurons)
        self.input_values = np.zeros(input_size)
        self.output_values = np.zeros(number_of_neurons)
    
    def sum_input(self,x):
        
        return np.matmul(x, self.weights)+self.biases
    
    def glorot_uniform(self, shape):
        
        limit = np.sqrt(6/(self.input_size+self.number_neurons))
        
        return np.random.random_sample(shape)*limit -limit
    
    def sigmoid_activation(self,x):
        
        return expit(x)
    
    def foward(self,x):
        
        # Foward Pass
        self.input_values = x
        self.output_values = self.sum_input(x)
        
        if self.activation == "sigmoid":
            self.output_values = self.sigmoid_activation(self.output_values)
        
        
        return self.output_values
    
    def backpropagation_wrt_weights(self, loss):
        
        # Calculates derivative of the activation function (sigmoid)
        d_loss = loss* self.compute_loss_wrt_activation()
        
        # Computing loss with regard to each weight
        self.weight_loss = np.zeros((self.input_size, self.number_neurons))
        
        for i in range(self.input_size):
            for j in range(self.number_neurons):
                self.weight_loss[i][j] = d_loss[j]*self.input_values[i]
        
        # Computing loss with regard to each bias
        self.bias_loss = d_loss
        
        return self.weight_loss, self.bias_loss
        
    def backpropagation_wrt_layers(self,loss):
        
        # Calculates derivative of the activation function (sigmoid)
        d_loss = loss* self.compute_loss_wrt_activation()
        
        # Computing loss with regard to each input
        self.input_loss = np.zeros(self.input_size)
        
        for i in range(self.input_size):
            loss_sum = 0
            for j in range(self.number_neurons):
                loss_sum += d_loss[j]*self.weights[i][j]
            self.input_loss[i] = loss_sum
        
        return self.input_loss
    
    def compute_loss_wrt_activation(self):
        
        d_activation = np.full(self.number_neurons, 1)
        
        if self.activation == "sigmoid":
            d_activation = self.output_values*(1-self.output_values)
        
        return d_activation
        
    def apply_gradients(self, lr = 0.001):
        
        # Applies gradient descent to the weights
        for i in range(self.input_size):
            for j in range(self.number_neurons):
                self.weights[i][j] = self.weights[i][j] - lr*self.weight_loss[i][j]
                
        # Applies gradient descent to the biases:
        for i in range(self.number_neurons):
            self.biases[i] = self.biases[i] - lr* self.bias_loss[i]
    


class NN_Network:
    
    def __init__(self, input_size, output_size, number_of_layers = 3, neurons_per_layer = 30, activation = "sigmoid", loss = "binary_cross_entropy"):
        
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.nn_output = np.zeros(output_size)
        self.loss = loss
        self.total_loss = 0
        
        # Adding layers
        self.layers.append(NN_Inside_Layer(input_size, number_of_neurons = neurons_per_layer))
        for _ in range(number_of_layers-2):
            self.layers.append(NN_Inside_Layer(neurons_per_layer, number_of_neurons = neurons_per_layer))
        self.layers.append(NN_Inside_Layer(neurons_per_layer, number_of_neurons = output_size, activation = activation))
         
    def predict(self, x):
        
        assert len(x) == self.input_size
        
        # Foward Pass
        y = x
        for layer in self.layers:
            y = layer.foward(y)        
        self.nn_output = y
        
        return y
    
    #### MSE LOSS ####
    def compute_mse_loss_derivate(self, y):
        
        assert len(y) == self.output_size
        
        # Computing loss
        d_loss = 2*(self.nn_output - y)
        #print(loss)
        
        return d_loss
    
    def compute_mse_loss(self, y):
        
        assert len(y) == self.output_size
        
        # Computing loss
        loss = np.power((self.nn_output - y), 2)
        #print(loss)
        
        return loss
        
    def apply_loss(self, loss):
        # Backward pass
        backward_pass = loss
        
        for layer in self.layers[::-1]:
            layer.backpropagation_wrt_weights(backward_pass)
            backward_pass = layer.backpropagation_wrt_layers(backward_pass)
            layer.apply_gradients()
        
    def compute_categorical_cross_entropy_loss(self, y):
    
        assert len(y) == self.output_size
        
        # Computing CC loss
        loss = 0
        
        for i in range(len(y)):
            if y[i] ==1:
                loss = loss +(-np.log(self.nn_output[i]))
            else:
                loss = loss+ (-np.log(1-self.nn_output[i]))
        
        return loss
    
    def compute_categorical_cross_entropy_loss_derivative(self, y):
    
        assert len(y) == self.output_size
        
        # Computing CC loss derivative
        d_loss = np.zeros(self.output_size)
        
        for i in range(len(y)):
            if y[i] ==1:
                d_loss[i] = -(1/self.nn_output[i])
            else:
                d_loss[i] = -(1/(self.nn_output[i]-1))
        
        return d_loss
    
    def compute_binary_cross_entropy_loss(self, y):
        
        assert len(y) ==1
        
        if y[0] == 1:
            return -np.log(self.nn_output)
        
        else:
            return -np.log(1 - self.nn_output)
    
    def compute_binary_cross_entropy_loss_derivate(self, y):
        
        assert len(y) ==1
        
        if y[0]==0:
            return 1/(1-self.nn_output)
        
        else:
            return -1/self.nn_output
    
    def fit(self, x, y):
        
        self.predict(x)
        d_loss = np.zeros(self.output_size)
        self.total_loss = 0
        
        if self.loss == "categorical_cross_entropy":
            d_loss = self.compute_categorical_cross_entropy_loss_derivative(y)
            self.total_loss = self.compute_categorical_cross_entropy_loss(y)/self.output_size
        elif self.loss == "mse":
            d_loss = self.compute_mse_loss_derivate(y)
            self.total_loss = self.compute_mse_loss(y).sum()/self.output_size
        elif self.loss == "binary_cross_entropy":
            d_loss = self.compute_binary_cross_entropy_loss_derivate(y)
            self.total_loss = self.compute_binary_cross_entropy_loss(y)
        else:
            raise Exception("loss not specified correctly")
        
        
        self.apply_loss(d_loss)
        return self.total_loss