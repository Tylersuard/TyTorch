import numpy as np

class Network:
    def __init__(self, list_of_neurons_by_layer):
        self.network_layers = {}
        for i in range(len(list_of_neurons_by_layer)):
            if i== 0:
                continue
            this_layer_dict = {}
            layer_weights = np.random.random_sample((list_of_neurons_by_layer[i-1],list_of_neurons_by_layer[i]))
            this_layer_dict["weights"] = layer_weights
            layer_biases = np.ones((list_of_neurons_by_layer[i],1))
            this_layer_dict["biases"] = layer_biases
            if i == len(list_of_neurons_by_layer)-1:
                layer_name = "Output Layer"
            else:
                layer_name = f"Hidden Layer {i}"
            self.network_layers[layer_name] = this_layer_dict

    def activation_function(self, output):
        return max(output,0)

    def forward_pass(self, input_data):
        for network_layer in self.network_layers:
            weighted_input = np.matmul(self.network_layers[network_layer]["weights"].T, input_data)
            biased_input = weighted_input + self.network_layers[network_layer]["biases"]
            activated_output = np.vectorize(self.activation_function)

            input_data = activated_output(biased_input)
            self.network_layers[network_layer]["layer_outputs"] = input_data
        network_output = input_data
        return network_output
    
    def calculate_loss(self, desired_output, actual_output):
        loss_vector = desired_output - actual_output
        loss_vector = loss_vector **2
        loss_total = np.sum(loss_vector)
        return loss_total
    
    def backpropagate(self, loss, desired_outputs, data_inputs):
        learning_rate = 0.0002

    #Set up a way to call the individual network layers
        network_layers = self.network_layers
        layer_names_list = list(network_layers.keys())

    #Get the derivatives of the loss wrt the network's outputs (AKA the Output Signal)
        last_layer = network_layers[layer_names_list[-1]]
        last_layer_derivatives_of_loss_wrt_outputs =  np.ones(last_layer["layer_outputs"].shape)

        for o in range(last_layer["layer_outputs"].shape[0]):
            focused_output = last_layer["layer_outputs"][o,0]
            corresponding_desired_output = desired_outputs[o,0]
            derivative_of_loss_wrt_output = 2*(last_layer["layer_outputs"][o,0]) - 2*(desired_outputs[o,0])
            last_layer_derivatives_of_loss_wrt_outputs[o,0] = derivative_of_loss_wrt_output
        last_layer["output_signal"] =  last_layer_derivatives_of_loss_wrt_outputs

    #That was the only unique thing you needed from the last layer, now just loop through the other layers.
        layer_countdown_list = reversed(range(len(layer_names_list)))
        for p in layer_countdown_list:

        #Set current and upstream layer names
            current_layer_name = layer_names_list[p]
            upstream_layer_name = layer_names_list[p-1]
        #Get current and upstream layers
            current_layer = network_layers[current_layer_name]

            if upstream_layer_name == "Output Layer":
                upstream_layer_name = None
                upstream_layer = None
            else:
                upstream_layer = network_layers[upstream_layer_name]
        #If there is no downstream layer, just use the Output Signal instead of the downstream error gradient
            try:
                downstream_layer_name = layer_names_list[p+1]
                downstream_layer = network_layers[downstream_layer_name]
                downstream_error_gradient = downstream_layer["error_gradient"]
            except:
                downstream_error_gradient = last_layer["output_signal"]
                downstream_layer_name = None

            #Find the activation gradient
            current_layer_drvs_of_outputs_wrt_inputs = np.ones(current_layer['layer_outputs'].shape)
            for k in range(current_layer["layer_outputs"].shape[0]):
                focused_output = current_layer["layer_outputs"][k,0]
                if focused_output > 0:
                    drv_output_wrt_input = 1
                else:
                    drv_output_wrt_input = 0
                current_layer_drvs_of_outputs_wrt_inputs[k,0] = drv_output_wrt_input


            current_layer["activation_gradient"] = np.multiply(downstream_error_gradient, current_layer_drvs_of_outputs_wrt_inputs)

            #find the weight gradient
            if upstream_layer is None:
                derivatives_of_inputs_wrt_weights = np.tile(data_inputs, (1,current_layer["weights"].shape[1]))                
            else:
                derivatives_of_inputs_wrt_weights = np.tile(upstream_layer["layer_outputs"],(1,current_layer["weights"].shape[1]))
            derivative_of_loss_wrt_inputs = current_layer["activation_gradient"]
            derivative_of_weights_wrt_loss = derivatives_of_inputs_wrt_weights * derivative_of_loss_wrt_inputs.T
            current_layer["weight_gradient"] = derivative_of_weights_wrt_loss

            #Find the bias gradient
            derivative_of_biases_wrt_loss = current_layer["activation_gradient"] * 1
            current_layer["bias_gradient"] = derivative_of_biases_wrt_loss

            #Find the error gradient
            derivatives_of_upstream_layer_outputs_wrt_current_layer_inputs = current_layer["weights"]
            error_gradient = np.dot(derivatives_of_upstream_layer_outputs_wrt_current_layer_inputs, current_layer["activation_gradient"])
            current_layer["error_gradient"] = error_gradient

            network_layers[current_layer_name] = current_layer
        
        #apply all the gradients times the learning rate
        for q in range(len(layer_names_list)):
            current_layer_name = layer_names_list[q]
            current_layer = network_layers[current_layer_name]

            weight_update = current_layer["weight_gradient"] *learning_rate
            updated_weights = current_layer["weights"] - weight_update
            current_layer["weights"] = updated_weights

            bias_update = current_layer["bias_gradient"] * learning_rate
            updated_biases = current_layer["biases"] - bias_update
            current_layer["biases"] = updated_biases

            self.network_layers[current_layer_name] = current_layer
        
        return self

network = Network([5,10,10,3])

input_data = np.random.random_sample((5,1))
desired_outputs = np.random.random_sample((3,1))

epochs = 3
for i in range(epochs):
    network_output = network.forward_pass(input_data)
    loss = network.calculate_loss(desired_outputs,network_output)
    print(f"Loss: {loss}")
    network = network.backpropagate(loss,desired_outputs, input_data)
