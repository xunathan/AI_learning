#include<iostream>
#include<fstream>
#include<vector>
#include<cuda_runtime_api.h>
#include<NvInfer.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

int main() {
    // Load the serialized engine from file
    std::ifstream engineFile("model.engine", std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error opening engine file!" << std::endl;
        return -1;
    }
    engineFile.seekg(0, engineFile.end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    // Create the TensorRT runtime and deserialize the engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);

    // Create an execution context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // Allocate memory for input and output buffers
    float* inputBuffer;
    float* outputBuffer;
    cudaMalloc(&inputBuffer, sizeof(float) * 3 * 224 * 224); // Example input size
    cudaMalloc(&outputBuffer, sizeof(float) * 1000); // Example output size

    // Prepare input data (this is just a placeholder)
    std::vector<float> inputData(3 * 224 * 224, 0.0f); // Fill with zeros for example
    cudaMemcpy(inputBuffer, inputData.data(), sizeof(float) * inputData.size(), cudaMemcpyHostToDevice);

    // Execute inference
    void* buffers[] = {inputBuffer, outputBuffer};
    context->executeV2(buffers);

    // Retrieve output data
    std::vector<float> outputData(1000);
    cudaMemcpy(outputData.data(), outputBuffer, sizeof(float) * outputData.size(), cudaMemcpyDeviceToHost);

    std::cout << "Inference completed. Output size: " << outputData.size() << std::endl;
    std::cout << "Top 5 output value: ";

    for (int i = 0; i < 5; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(inputBuffer);
    cudaFree(outputBuffer);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}