#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <cstdlib>
#include <vector>

std::vector<std::vector<int>> images;
std::vector<int> labels;

bool loadMNIST(const char* filename) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        printf("Error opening file: %s\n", filename);
        return false;
    }

    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);  // go back to beginning

    // read the whole file into buffer
    char *buffer = (char*) malloc(fileSize);
    fread(buffer, 1, fileSize, f);

    // skipping header line
    int found = -1;
    int numLines = 0;
    for (long i = 0; i < fileSize; i++) {
        if (buffer[i] == '\n') {  
            if (found == -1) {
                found = i;
            }
            numLines++;
        }
    }
    printf("Number of lines (excluding header): %d\n", numLines);

    images.clear();
    labels.clear();
    
    int currRow = 0;
    images.push_back(std::vector<int>());  // create first row
    
    // read first label
    long i = found + 1;
    labels.push_back(buffer[i] - '0');
    i += 2;  // skip label and comma

    while (i < fileSize) {
        if (buffer[i] == '\n') {
            i++;
            if (i >= fileSize) break;  
            currRow++;
            images.push_back(std::vector<int>());
            // read label for this new row
            char labelChar = buffer[i];
            labels.push_back(labelChar - '0');
            i++;  // move past label
            i++;  // skip comma after label
            if (i >= fileSize) break;  
        }

        // Parse pixel value
        char pixel[8];  // Increased size to handle larger numbers safely
        int j = 0;
        while(i < fileSize && buffer[i] != ',' && buffer[i] != '\n' && j < 7) {
            pixel[j++] = buffer[i];
            i++;
        }
        pixel[j] = '\0';
        
        int pixelValue = atoi(pixel);
        images[currRow].push_back(pixelValue);

        // skip comma if present
        if (i < fileSize && buffer[i] == ',') {
            i++;
        }
    }

    free(buffer);
    fclose(f);
    return true;
}

#endif // MNIST_H