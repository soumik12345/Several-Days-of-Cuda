#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <cstring>


__global__ void rgb_to_gray(uchar3* dataIn, unsigned char* dataOut, uint imageHeight, uint imageWidth) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex < imageWidth && yIndex < imageHeight) {
        uchar3 rgb = dataIn[yIndex * imageWidth + xIndex];
        dataOut[yIndex * imageWidth + xIndex] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}


class RgbToGray {

public:

    int arrayLength;

    explicit RgbToGray(int arrayLength);

    void initArray(int* array) const;
    void run(char *sourceImageFile) const;
    void displayResult(int* array, int* resultArray) const;
};

RgbToGray::RgbToGray(int arrayLength) {
    this->arrayLength = arrayLength;
}

void RgbToGray::initArray(int *array) const {
    for(int i = 0; i < this->arrayLength; i++)
        array[i] = rand() % 100;
}

void RgbToGray::displayResult(int *array, int* resultArray) const {
    for(int i = 0; i < this->arrayLength; i++)
        printf("%d * 2 = %d\n", array[i], resultArray[i]);
}

void RgbToGray::run(char *sourceImageFile) const {

    int deviceId = cudaGetDevice(&deviceId);

    printf("GPU Device ID: %d\n", deviceId);
    printf("CPU Device ID: %d\n\n", cudaCpuDeviceId);

    cv::Mat sourceImage = cv::imread(sourceImageFile);
    const uint imageHeight = sourceImage.rows, imageWidth = sourceImage.cols;
    cv::Mat grayImage(imageHeight, imageWidth, CV_8UC1, cv::Scalar(0));

    uchar3 * dataIn;
    unsigned char * dataOut;

    cudaMalloc((void**)&dataIn, imageHeight * imageWidth * sizeof(uchar3));
    cudaMalloc((void**)&dataOut, imageHeight * imageWidth * sizeof(unsigned char));

    cudaMemcpy(
            dataIn, sourceImage.data,
            imageHeight * imageWidth * sizeof(uchar3),
            cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
            (imageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (imageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgb_to_gray<<<blocksPerGrid, threadsPerBlock>>>(dataIn, dataOut, imageHeight, imageWidth);
    cudaDeviceSynchronize();

    cudaMemcpy(
            grayImage.data, dataOut,
            imageHeight * imageWidth * sizeof(unsigned char),
            cudaMemcpyDeviceToHost);

    cudaFree(dataIn);
    cudaFree(dataOut);

    cv::imwrite("./output,jpg", grayImage);
}


int main() {
    RgbToGray program(16);
    program.run("../../../assets/color_image_1.jpg");
}
