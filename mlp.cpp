#include <stdio.h>
#include <cstdlib>  
#include <ctime>
#include <cmath>
#include "mnist.h" 
#include <random>

enum ActivationFunction {
    SIGMOID,
    RELU,
    SOFTMAX
};

struct Neuron {
    float value; // output after activation
    float bias;
    float* weights;
    
    float delta; // dl / dz (preactivation)
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


float randomWeight(int fan_in) {
    static std::default_random_engine gen(std::random_device{}());
    float stddev = sqrtf(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, stddev);
    return dist(gen);
}


Layer createLayer(int neuronCount, int inputPerNeuron, ActivationFunction activation = SIGMOID) {
    Layer layer;
    layer.neuronCount = neuronCount;
    layer.neurons = new Neuron[neuronCount];
    layer.inputSize = inputPerNeuron;
    layer.layerInputs = new float[inputPerNeuron];
    layer.activation = activation;

    for (int i = 0; i < neuronCount; i++) {
        layer.neurons[i].bias = 0.0f;
        layer.neurons[i].weights = new float[inputPerNeuron];
        
        layer.neurons[i].delta = 0.0f;
        
        for (int j = 0; j < inputPerNeuron; j++) {
            layer.neurons[i].weights[j] = randomWeight(inputPerNeuron);
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
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
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
                // dL / d z this layer 
                layer.neurons[i].delta = sumDeltas * sigmoidDerivative;
                
            }

        }
    }

    for(int l = 0; l < mlp.layerCount; l++) {
        Layer& layer = mlp.layers[l];
        for (int i = 0; i < layer.neuronCount; i++) {
            for (int j = 0; j < layer.inputSize; j++) {
                float gradient = layer.neurons[i].delta * layer.layerInputs[j]; // dL / d weight
                layer.neurons[i].weights[j] -= learningRate * gradient;
            }
            layer.neurons[i].bias -= learningRate * layer.neurons[i].delta;
        }
    }
}


int main() {
    srand(time(0));  
    
    printf("Loading MNIST data...\n");
    if (!loadMNIST("mnist_train.csv")) {
        printf("Failed to load MNIST data\n");
        return 1;
    }

    
    printf("Loaded %zu images\n", images.size());
    
    int layerSizes[] = {128, 64, 10}; 
    ActivationFunction activations[] = {SIGMOID, SIGMOID, SOFTMAX};
    int layerCount = sizeof(layerSizes) / sizeof(layerSizes[0]);
    int inputSize = 784; 
    
    int max = 0;
    int min = 255;
    for (int i = 0; i < images.size(); i++) {
        for(int j = 0; j < inputSize; j++) {
            if (images[i][j] > max) {
                max = images[i][j];
            }
            if (images[i][j] < min) {
                min = images[i][j];
            }
            
        }
    }

    printf("Max pixel value across all images: %d\n", max);
    printf("Min pixel value across all images: %d\n", min);

    

    MLP mlp = createMLP(layerSizes, layerCount, inputSize, activations);
    
    float learningRate = 0.01f;
    
    printf("Training...\n");
    
    for (int epoch = 0; epoch < 10; epoch++) {
        clock_t startTime = clock();
        
        for (size_t i = 0; i < images.size(); i++) {

            // Normalize pixel values to [0, 1]
            float* normalizedInput = new float[inputSize];
            for (int j = 0; j < inputSize; j++) {
                normalizedInput[j] = images[i][j] / 255.0f;
            }
            
            // one hot encoded target
            float target[10] = {0.0f};
            target[labels[i]] = 1.0f;
            
            float output[10];
            forwardMLP(mlp, normalizedInput, inputSize, output);
            backwardMLP(mlp, normalizedInput, inputSize, output, target, learningRate);
            
            delete[] normalizedInput;
        }
        
        clock_t endTime = clock();
        double elapsedTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
        printf("Epoch %d completed in %.2f seconds\n", epoch + 1, elapsedTime);
    }
    
    // Load test data
    printf("\nLoading test data...\n");
    if (!loadMNIST("mnist_test.csv")) {
        printf("Failed to load test data\n");
        return 1;
    }
    
    printf("Loaded %zu test images\n", images.size());
    
    int correct = 0;
    for (size_t i = 0; i < images.size(); i++) {
        float* normalizedInput = new float[inputSize];
        for (int j = 0; j < inputSize; j++) {
            normalizedInput[j] = images[i][j] / 255.0f;
        }
        
        float output[10];
        forwardMLP(mlp, normalizedInput, inputSize, output);
        
        int predicted = 0;
        for (int j = 1; j < 10; j++) {
            if (output[j] > output[predicted]) {
                predicted = j;
            }
        }
        
        if (predicted == labels[i]) {
            correct++;
        }
        

        if (i < 10) {
            printf("Sample %zu: Predicted: %d, Actual: %d | Outputs: [", i, predicted, labels[i]);
            for (int j = 0; j < 10; j++) {
                printf("%.4f", output[j]);
                if (j < 9) printf(", ");
            }
            printf("]\n");
        }
        
        delete[] normalizedInput;
    }
    
    printf("\nTest Accuracy: %d/%zu (%.2f%%)\n", 
           correct, images.size(), 100.0f * correct / images.size());
    
    return 0;
}