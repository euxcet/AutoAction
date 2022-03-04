#include <android/asset_manager_jni.h>
#include "ActionNet.h"

static ActionNet* g_actionnet = nullptr;
// static NanoDet* g_nanodet = nullptr;
static ncnn::Mutex lock;

static int input_width = 128;
static int input_height = 6;
static int input_channel = 1;
static int class_num = 2;


void loadActionNet(JNIEnv *env, const char* param_path, const char* bin_path, int numOfThread) {
    Logger("--- Init ActionNet ---\n");
    g_actionnet = new ActionNet();
    g_actionnet -> setNumThread(numOfThread);
    g_actionnet -> initModel(param_path, bin_path);
    Logger("--- ActionNet Detect ---\n");
}

void loadActionNetByAsset(JNIEnv *env, jobject assetManager, int numOfThread) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    Logger("--- Init ActionNet ---\n");
    g_actionnet = new ActionNet();
    g_actionnet -> setNumThread(numOfThread);
    g_actionnet -> initModel(mgr);
    Logger("--- ActionNet Detect ---\n");
}

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    LOGI("Goodbye OcrLite!");
    {
        ncnn::MutexLockGuard g(lock);
        delete g_actionnet;
        g_actionnet = nullptr;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_benjaminwan_ocrlibrary_NcnnFunction_initByAsset(JNIEnv *env, jobject thiz, jobject assetManager,
                                                  jint numThread) {
    loadActionNetByAsset(env, assetManager, 4);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_benjaminwan_ocrlibrary_NcnnFunction_print(JNIEnv *env, jobject thiz) {
    LOGE("TEST");
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_benjaminwan_ocrlibrary_NcnnFunction_actionDetect(JNIEnv *env, jobject thiz,
                                                          jfloatArray data) {
    if (g_actionnet != nullptr) {
        float *n_data = (float *) env->GetFloatArrayElements(data, 0);
        int result = g_actionnet->detect(n_data, input_width, input_height, input_channel, class_num);
        env->ReleaseFloatArrayElements(data, n_data, 0);
        return (jint) result;
    } else {
        return (jint) -1;
    }
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_benjaminwan_ocrlibrary_NcnnFunction_init(JNIEnv *env, jobject thiz, jstring param_path,
                                                  jstring bin_path, jint num_thread, int width, int height, int channel, int classNum) {
    loadActionNet(env, env -> GetStringUTFChars(param_path, 0),  env -> GetStringUTFChars(bin_path, 0), 4);
    input_width = width;
    input_height = height;
    input_channel = channel;
    class_num = classNum;
    return JNI_TRUE;
}