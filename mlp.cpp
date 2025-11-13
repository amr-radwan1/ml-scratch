#include <stdio.h>
#include <cstdlib>  
#include <ctime>
#include <cmath> 

struct Neuron {
    float value; // output after activation
    float bias;
    float* weights;
    
    float delta;       
};

struct Layer {
    Neuron* neurons;
    int neuronCount;
    
    float* layerInputs; // store inputs to layer for derivative in backprop
    int inputSize;
};

struct MLP {
    Layer* layers;
    int layerCount;
};

float randomWeight() {
    return ((float) rand() / RAND_MAX) * 2 - 1;
}

Layer createLayer(int neuronCount, int inputPerNeuron) {
    Layer layer;
    layer.neuronCount = neuronCount;
    layer.neurons = new Neuron[neuronCount];
    layer.inputSize = inputPerNeuron;
    layer.layerInputs = new float[inputPerNeuron];  

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
    return 1 / (1 + expf(-x));
}



void forwardLayer(Layer& layer, float* inputs, int inputSize, float* outputs) {
    for (int j = 0; j < inputSize; j++) {
        layer.layerInputs[j] = inputs[j];
    }
    
    for (int i = 0; i < layer.neuronCount; i++) {
        float activation = dotProduct(layer.neurons[i].weights, inputs, inputSize) + layer.neurons[i].bias;
        layer.neurons[i].value = sigmoid(activation);
        outputs[i] = layer.neurons[i].value;
    }
}

MLP createMLP(int* layerSizes, int layerCount, int inputSize) {
    MLP mlp;
    mlp.layerCount = layerCount;
    mlp.layers = new Layer[layerCount];

    for (int i = 0; i < layerCount; i++) {
        int prevLayerSize = (i == 0) ? inputSize : layerSizes[i - 1];
        mlp.layers[i] = createLayer(layerSizes[i], prevLayerSize);
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
        float sigmoidDerivative = output * (1.0f - output); // dy_pred / d output
        outputLayer.neurons[i].delta = error * sigmoidDerivative; // dL / d output

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
            float sigmoidDerivative = output * (1.0f - output);
            layer.neurons[i].delta = sumDeltas * sigmoidDerivative;

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
    int layerCount = 3;
    int inputSize = 2;
    
    MLP mlp = createMLP(layerSizes, layerCount, inputSize);
    
    float learningRate = 0.1f;

    for (int epoch = 0; epoch < 10000; epoch++) {
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