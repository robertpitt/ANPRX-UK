#include <jni.h>
#include <opencv2/core/core.hpp>


using namespace std;
using namespace cv;

extern "C" {
}
extern "C" {
JNIEXPORT void JNICALL Java_dev_robertpitt_anprX_opencv_NumberplateDetectorV2_detect_1c(
    JNIEnv *env,
    jclass clazz,
    jlong src_address) {
        // Get the Mat memory pointer
        Mat& src  = *(Mat*)src_address;

       src.release();
    }
}