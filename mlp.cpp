#include <stdio.h>
#include <cstdlib>  
#include <ctime>
#include <cmath> 

enum ActivationFunction {
    SIGMOID,
    RELU,
    SOFTMAX
};

struct Neuron {
    float value; // output after activation
    float bias;
    float* weights;
    
    float delta;
    float preActivation; // store z before activation for derivatives
};

struct Layer {
    Neuron* neurons;
    int neuronCount;
    
    float* layerInputs; // store inputs to layer for derivative in backprop
    int inputSize;
    
    ActivationFunction activation;
};

struct MLP {
    Layer* layers;
    int layerCount;
};




float randomWeight() {
    return ((float) rand() / RAND_MAX) * 2 - 1;
}

Layer createLayer(int neuronCount, int inputPerNeuron, ActivationFunction activation = SIGMOID) {
    Layer layer;
    layer.neuronCount = neuronCount;
    layer.neurons = new Neuron[neuronCount];
    layer.inputSize = inputPerNeuron;
    layer.layerInputs = new float[inputPerNeuron];
    layer.activation = activation;

    for (int i = 0; i < neuronCount; i++) {
        layer.neurons[i].bias = randomWeight();
        layer.neurons[i].weights = new float[inputPerNeuron];
        
        layer.neurons[i].delta = 0.0f;
        
        for (int j = 0; j < inputPerNeuron; j++) {
            layer.neurons[i].weights[j] = randomWeight();
        }
    }

    return layer;
}

float dotProduct(float* a, float* b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
        sum += a[i] * b[i];        
    return sum;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float reluDerivative(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

void softmax(float* input, int size, float* output) {
    float maxVal = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > maxVal) maxVal = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - maxVal);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}



void forwardLayer(Layer& layer, float* inputs, int inputSize, float* outputs) {
    for (int j = 0; j < inputSize; j++) {
        layer.layerInputs[j] = inputs[j];
    }
    
    for (int i = 0; i < layer.neuronCount; i++) {
        float activation = dotProduct(layer.neurons[i].weights, inputs, inputSize) + layer.neurons[i].bias;
        layer.neurons[i].preActivation = activation;
        if (layer.activation == RELU) {
            layer.neurons[i].value = relu(activation);
        } else if (layer.activation == SOFTMAX) {
            layer.neurons[i].value = activation;
        } else {
            layer.neurons[i].value = sigmoid(activation);
        }
        outputs[i] = layer.neurons[i].value;
    }
    
    if (layer.activation == SOFTMAX) {
        softmax(outputs, layer.neuronCount, outputs);
        for (int i = 0; i < layer.neuronCount; i++) {
            layer.neurons[i].value = outputs[i];
        }
    }
}

MLP createMLP(int* layerSizes, int layerCount, int inputSize, ActivationFunction* activations = nullptr) {
    MLP mlp;
    mlp.layerCount = layerCount;
    mlp.layers = new Layer[layerCount];

    for (int i = 0; i < layerCount; i++) {
        int prevLayerSize = (i == 0) ? inputSize : layerSizes[i - 1];
        ActivationFunction act = activations ? activations[i] : SIGMOID;
        mlp.layers[i] = createLayer(layerSizes[i], prevLayerSize, act);
    }

    return mlp;
}


void forwardMLP(MLP& mlp, float* input, int inputSize, float* output) {
    if (mlp.layerCount == 0) {
        for (int i = 0; i < inputSize; i++) {
            output[i] = input[i];
        }
        return;
    }

    float* currentInput = input;
    int currentInputSize = inputSize;

    float* layerOutput = nullptr;

    for (int i = 0; i < mlp.layerCount; i++) {
        Layer& layer = mlp.layers[i];
        layerOutput = new float[layer.neuronCount];

        forwardLayer(layer, currentInput, currentInputSize, layerOutput);

        if (i > 0) delete[] currentInput;
        currentInput = layerOutput;
        currentInputSize = layer.neuronCount;
    }

    for (int i = 0; i < mlp.layers[mlp.layerCount - 1].neuronCount; i++) {
        output[i] = layerOutput[i];
    }
    delete[] layerOutput;
}

void backwardMLP(MLP& mlp, float* input, int inputSize, float* predictions, float* targets, float learningRate = 0.01f) {
    Layer& outputLayer = mlp.layers[mlp.layerCount - 1];
    
    for (int i = 0; i < outputLayer.neuronCount; i++) {
        float output = outputLayer.neurons[i].value;
        float error = output - targets[i]; // dL / d y_pred
        if (outputLayer.activation == SOFTMAX) {
            outputLayer.neurons[i].delta = error; // dL / d output
        } else if (outputLayer.activation == RELU) {
            outputLayer.neurons[i].delta = error * reluDerivative(outputLayer.neurons[i].preActivation);
        } else {
            float sigmoidDerivative = output * (1.0f - output); // dy_pred / d output
            outputLayer.neurons[i].delta = error * sigmoidDerivative; // dL / d output
        }

        for (int j = 0; j < outputLayer.inputSize; j++) {
            // dL / d output * d output / d weight
            float gradient = outputLayer.neurons[i].delta * outputLayer.layerInputs[j]; // dL / d weight
            outputLayer.neurons[i].weights[j] -= learningRate * gradient;
        }

        outputLayer.neurons[i].bias -= learningRate * outputLayer.neurons[i].delta;
    }

    for (int l = mlp.layerCount - 2; l >= 0; l--) {
        Layer& layer = mlp.layers[l];
        Layer& nextLayer = mlp.layers[l + 1];

        for (int i = 0; i < layer.neuronCount; i++) {
            float sumDeltas = 0.0f;
            for (int k = 0; k < nextLayer.neuronCount; k++) {
                // dL / d nextLayer output * d nextLayer output / d this layer output
                sumDeltas += nextLayer.neurons[k].weights[i] * nextLayer.neurons[k].delta;
            }
            float output = layer.neurons[i].value;
            if (layer.activation == RELU) {
                layer.neurons[i].delta = sumDeltas * reluDerivative(layer.neurons[i].preActivation);
            } else {
                float sigmoidDerivative = output * (1.0f - output);
                layer.neurons[i].delta = sumDeltas * sigmoidDerivative;
            }

            for (int j = 0; j < layer.inputSize; j++) {
                float gradient = layer.neurons[i].delta * layer.layerInputs[j];
                layer.neurons[i].weights[j] -= learningRate * gradient;

            }

            layer.neurons[i].bias -= learningRate * layer.neurons[i].delta;
        }
    }

  
}


int main() {
    srand(time(0));  
    
    float xorInputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    

    float xorTargets[4][2] = {
        {1.0f, 0.0f},  
        {0.0f, 1.0f},  
        {0.0f, 1.0f}, 
        {1.0f, 0.0f}   
    };
    
    int layerSizes[] = {8, 8, 2}; 
    ActivationFunction activations[] = {RELU, RELU, SOFTMAX};
    int layerCount = 3;
    int inputSize = 2;
    
    MLP mlp = createMLP(layerSizes, layerCount, inputSize, activations);
    
    float learningRate = 0.01f;

    for (int epoch = 0; epoch < 500; epoch++) {
        for (int i = 0; i < 4; i++) {
            float output[2];
            forwardMLP(mlp, xorInputs[i], inputSize, output);
            backwardMLP(mlp, xorInputs[i], inputSize, output, xorTargets[i], learningRate);
        }
    }
    for (int i = 0; i < 4; i++) {
        float output[2];
        forwardMLP(mlp, xorInputs[i], inputSize, output);
        printf("Input: [%.1f, %.1f] Output: [%.4f, %.4f], Target: [%.1f, %.1f]\n", 
               xorInputs[i][0], xorInputs[i][1], 
               output[0], output[1],
               xorTargets[i][0], xorTargets[i][1]
               );
    }
    
    
    return 0;

}