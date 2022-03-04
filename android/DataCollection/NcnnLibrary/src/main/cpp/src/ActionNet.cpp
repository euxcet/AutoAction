#include "ActionNet.h"
#include <stdlib.h>

ActionNet::~ActionNet() {
    net.clear();
}

void ActionNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

bool ActionNet::initModel(AAssetManager *mgr) {
    int ret_param = net.load_param(mgr, "best.param");
    int ret_bin = net.load_model(mgr, "best.bin");

    if (ret_param != 0 || ret_bin != 0) {
        LOGE("ActionNet # %d  %d", ret_param, ret_bin);
        return false;
    }
    return true;
}

bool ActionNet::initModel(const char* param_path, const char* bin_path) {
    int ret_param = net.load_param(param_path);
    int ret_bin = net.load_model(bin_path);

    if (ret_param != 0 || ret_bin != 0) {
        LOGE("ActionNet # %d  %d", ret_param, ret_bin);
        return false;
    }
    return true;
}

int ActionNet::detect(float* data, int input_width, int input_height, int input_channel, int class_num) {
    ncnn::Mat input(input_width * input_height * input_channel, data);
    input = input.reshape(input_width, input_height, input_channel);

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(numThread);
    extractor.input("input", input);
    ncnn::Mat out;
    extractor.extract("output", out);
    float *result = (float*)out.data;
    int pos = 0;
    float maxValue = -1e5;
    for (int i = 0; i < class_num; i++) {
        if (result[i] > maxValue) {
            maxValue = result[i];
            pos = i;
        }
    }
    return pos;
}