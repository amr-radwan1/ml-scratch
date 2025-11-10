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

float relu(float x) {
    return (x > 0) ? x : 0.0f;
}

void softmax(float* input, int size, float* output) {
    float sum = 0.0f;

    for(int i = 0; i < size; i++) {
        output[i] = expf(input[i]);  
        sum += output[i];
    }

    for(int i = 0; i < size; i++) {
        output[i] = output[i] / sum;
    }
}

void forwardLayer(Layer& layer, float* inputs, int inputSize, float* outputs, bool applyRelu) {
    for (int i = 0; i < layer.neuronCount; i++) {
        float activation = dotProduct(layer.neurons[i].weights, inputs, inputSize) + layer.neurons[i].bias;
        layer.neurons[i].value = applyRelu ? relu(activation) : activation;
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

        bool applyRelu = (i < mlp.layerCount - 1);
        forwardLayer(layer, currentInput, currentInputSize, layerOutput, applyRelu);

        if (i > 0) delete[] currentInput;
        currentInput = layerOutput;
        currentInputSize = layer.neuronCount;
    }

    softmax(layerOutput, mlp.layers[mlp.layerCount - 1].neuronCount, output);

    delete[] layerOutput;
}

void loadWeights(MLP& mlp) {
    float fc1_weights[8][2] = {
        {-0.289758f, -0.341226f},
        {1.443016f, -1.294722f},
        {-1.199916f, -1.235250f},
        {0.946683f, 0.788452f},
        {0.302699f, 0.209988f},
        {1.030959f, 0.869180f},
        {0.149584f, -0.129121f},
        {0.679692f, 0.972907f}
    };
    float fc1_bias[8] = {-0.352048f, -0.450067f, 1.223930f, -0.783500f, 
                         -0.671987f, -0.898360f, -0.676135f, 0.832687f};
    
    for (int i = 0; i < 8; i++) {
        mlp.layers[0].neurons[i].bias = fc1_bias[i];
        for (int j = 0; j < 2; j++) {
            mlp.layers[0].neurons[i].weights[j] = fc1_weights[i][j];
        }
    }
    
    float fc_hidden_weights[8][8] = {
        {0.164999f, -0.216283f, 0.267871f, 0.083952f, 0.086033f, -0.185758f, 0.265139f, -0.106266f},
        {0.058458f, 0.458599f, -1.138265f, -1.249847f, -0.322268f, -1.103818f, 0.141496f, 0.967665f},
        {0.052247f, 1.435651f, -1.633166f, -0.553701f, -0.093960f, -1.124455f, -0.266096f, 1.097304f},
        {0.229468f, -0.459296f, 1.602662f, 1.239617f, 0.297255f, 1.716253f, -0.134797f, 0.333224f},
        {-0.178098f, -0.111790f, -0.046482f, -0.156621f, -0.124190f, -0.314205f, -0.298897f, -0.015613f},
        {0.235458f, -0.223764f, 1.773395f, 1.503393f, 0.350309f, 1.497746f, 0.001594f, 0.298486f},
        {0.194882f, -0.371485f, -0.132792f, -0.105630f, -0.315042f, 0.302747f, -0.256041f, -0.233207f},
        {-0.104823f, -0.291684f, -0.253147f, 0.213346f, -0.091313f, -0.032163f, -0.326580f, -0.092474f}
    };
    float fc_hidden_bias[8] = {-0.274183f, 0.425024f, 0.974474f, -0.147347f,
                               -0.076183f, -0.191622f, -0.298609f, -0.270596f};
    
    for (int i = 0; i < 8; i++) {
        mlp.layers[1].neurons[i].bias = fc_hidden_bias[i];
        for (int j = 0; j < 8; j++) {
            mlp.layers[1].neurons[i].weights[j] = fc_hidden_weights[i][j];
        }
    }
    
    float fc2_weights[2][8] = {
        {-0.107642f, -1.142470f, -1.374395f, 1.812516f, 0.210806f, 1.780124f, 0.018185f, -0.247536f},
        {0.275798f, 0.934246f, 1.696588f, -1.516321f, -0.027512f, -1.548799f, -0.190009f, -0.289626f}
    };
    float fc2_bias[2] = {-0.871736f, 0.837122f};
    
    for (int i = 0; i < 2; i++) {
        mlp.layers[2].neurons[i].bias = fc2_bias[i];
        for (int j = 0; j < 8; j++) {
            mlp.layers[2].neurons[i].weights[j] = fc2_weights[i][j];
        }
    }
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