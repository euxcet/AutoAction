#ifndef __ACTIONNET_H__
#define __ACTIONNET_H__

#include <net.h>

#define TAG "ActionNet"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE,TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)

#define __ENABLE_CONSOLE__ true
#define Logger(format, ...) {\
  if(__ENABLE_CONSOLE__) LOGI(format,##__VA_ARGS__); \
}

class ActionNet {
public:
    ~ActionNet();

    void setNumThread(int numOfThread);

    bool initModel(AAssetManager *mgr);

    bool initModel(const char* param_path, const char* bin_path);

    int detect(float* data, int input_width, int input_height, int input_channel, int class_num);

private:
    int numThread;
    ncnn::Net net;

};


#endif