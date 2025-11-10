#include <stdio.h>
#include <cstdlib>  
#include <ctime>
#include <cmath> 

struct Neuron {
    float value;
    float bias;
    float* weights;
};

struct Layer {
    Neuron* neurons;
    int neuronCount;
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


    for (int i = 0; i < neuronCount; i++) {
        layer.neurons[i].bias = randomWeight();
        layer.neurons[i].weights = new float[inputPerNeuron];
        
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


int main() {
    float xorInputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    
    int layerSizes[] = {8, 8, 2}; 
    int layerCount = 3;
    int inputSize = 2;
    
    MLP mlp = createMLP(layerSizes, layerCount, inputSize);
    
 
    float output[2];
    for (int i = 0; i < 4; i++) {
        forwardMLP(mlp, xorInputs[i], inputSize, output);
        printf("Input: [%.1f, %.1f] -> [%.6f, %.6f] \n", 
               xorInputs[i][0], xorInputs[i][1], 
               output[0], output[1]
               );
    }
    
    return 0;

}