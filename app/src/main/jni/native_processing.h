//
// Created by Jose Honorato on 10/12/15.
//

#ifndef OPENCV_SAMPLE_NATIVE_PROCESSING_H
#define OPENCV_SAMPLE_NATIVE_PROCESSING_H

#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include "opencv2/ml/ml.hpp"


#include <vector>
#include <string>

extern "C" {
JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_FindFeatures(JNIEnv *, jobject,
                                                                                jlong addrGray,
                                                                                jlong addrRgba);
}
#endif //OPENCV_SAMPLE_NATIVE_PROCESSING_H
