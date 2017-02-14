//
// Created by Jose Honorato on 10/12/15.
//

#include "native_processing.h"

using namespace std;
using namespace cv;

extern "C" {
JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba) {
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;
    vector<KeyPoint> v;

    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mGr, v);
    for (unsigned int i = 0; i < v.size(); i++) {
        const KeyPoint& kp = v[i];
        circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }
}

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_IncreaseContrast(JNIEnv*, jobject, jlong addrRgba) {
    Mat& src = *(Mat*)addrRgba;
    Mat dst;

    // Convert to gray
    cvtColor(src, src, CV_BGR2GRAY);

    // Histogram equalization
    equalizeHist(src, dst);

    // Saturation by 10%
    float alpha = 1.1f;
    float beta = 12.75f;
    dst.convertTo(dst, -1, alpha, beta);
}

JNIEXPORT jstring JNICALL Java_org_honorato_opencvsample_MainActivity_stringFromJNI(JNIEnv *env,
                                                                                    jobject obj,
                                                                                    jstring input,
                                                                                    jintArray nums) {
    //const char * path;
    //path = const_cast<char*> ( env->GetStringUTFChars(input , NULL ) );
    //path = (env)->GetStringUTFChars( input , NULL ) ;
    //std:: string temp = path;


    const char *nativeString = env->GetStringUTFChars(input, 0);
    jint *c_array;
    c_array = env->GetIntArrayElements(nums, 0);
    c_array[0] = 5001;
    c_array[1] = 2;
    c_array[2] = 3;
    string temp = nativeString;
    std::string str = "Hi Danial From JNI" + temp;
    env->ReleaseStringUTFChars(input, nativeString);
    env->ReleaseIntArrayElements(nums, c_array, 0);
    return env->NewStringUTF(str.c_str());

}

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_CreateMask(JNIEnv *env, jobject,
                                                                              jlong addrRgba,
                                                                              jlong addrRgba_prev,
                                                                              jlong addrGr_output) {

    Mat &mRgb_prev = *(Mat *) addrRgba_prev;
    Mat &mRgb = *(Mat *) addrRgba;
    Mat &output = *(Mat *) addrGr_output;


    Mat mGr, mGr_prev;

    cvtColor(mRgb, mGr, CV_BGR2GRAY);
    cvtColor(mRgb_prev, mGr_prev, CV_BGR2GRAY);

    vector<KeyPoint> kp;
    vector<KeyPoint> kp_prev;


    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mGr_prev, kp_prev);
    /////////////////////////////////////
    //////////////////////////////////
    vector<Point2f> valid_points_prev, valid_points;
    bool optical_flow = true;
    const float inlier_th = 2.5f;

    if (optical_flow == true) {


        vector<Point2f> points;
        vector<Point2f> points_prev;

        for (int i = 0; i < kp_prev.size(); i++) {
            points_prev.push_back(kp_prev[i].pt);
        }

        vector<uchar> founded_points;
        Mat err;
        if (points_prev.size() > 0) {
            calcOpticalFlowPyrLK(mGr_prev, mGr, points_prev, points, founded_points, err,
                                 Size(25, 25), 2);

        }
        for (unsigned int i = 0; i < points_prev.size(); i++) {

            if (founded_points[i]) {
                //line(mRgb,points_prev[i],points[i],Scalar(0,255,0,255));
                //circle(mRgb,  points_prev[i], 10, Scalar(0,255,0,255));
                valid_points_prev.push_back(points_prev[i]);
                valid_points.push_back(points[i]);


            }

        }
        err.release();
    }


    Mat H;
    if (valid_points.size() > 10) {
        H = findHomography(valid_points_prev, valid_points, CV_RANSAC);
        Mat mask = Mat::zeros(mRgb.size(), CV_8U);
        // perspectiveTransform(mGr_prev,)

        Mat mGr_transformed;
        warpPerspective(mGr_prev, mGr_transformed, H, mGr_prev.size());
        //warpPerspective(mRgb_prev,mGr_transformed,H,mRgb_prev.size());

        Mat sub;
        absdiff(mGr, mGr_transformed, sub);

        sub = sub > 80;
        Mat squ = Mat::ones(2, 2, CV_8U);
        Mat big_rect = Mat::ones(20, 60, CV_8U);


        Mat h_line = Mat::ones(4, 80, CV_8U);
        Mat v_line = Mat::ones(20, 2, CV_8U);

        erode(sub, sub, squ);
        dilate(sub, sub, Mat::ones(2, 4, CV_8UC1));



        normalize(sub, sub, 0, 126, NORM_MINMAX);
        // sub.convertTo(sub,CV_32FC1);

        sub.copyTo(output);




        sub.release();
        mGr_transformed.release();
        mask.release();
        squ.release();
        h_line.release();
        v_line.release();
    }

    H.release();
    mGr.release();
    mGr_prev.release();

    return;
}

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_CreateMask2(JNIEnv *env, jobject,
                                                                              jlong addrRgba,
                                                                              jlong addrRgba_prev,
                                                                              jlong addrGr_output) {

    Mat &mRgb_prev = *(Mat *) addrRgba_prev;
    Mat &mRgb = *(Mat *) addrRgba;
    Mat &output = *(Mat *) addrGr_output;


    Mat mGr, mGr_prev;

    cvtColor(mRgb, mGr, CV_BGR2GRAY);
    cvtColor(mRgb_prev, mGr_prev, CV_BGR2GRAY);

    vector<KeyPoint> kp;
    vector<KeyPoint> kp_prev;


    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mGr_prev, kp_prev);
    /////////////////////////////////////
    //////////////////////////////////
    vector<Point2f> valid_points_prev, valid_points;
    bool optical_flow = true;
    const float inlier_th = 2.5f;

    if (optical_flow == true) {


        vector<Point2f> points;
        vector<Point2f> points_prev;

        for (int i = 0; i < kp_prev.size(); i++) {
            points_prev.push_back(kp_prev[i].pt);
        }

        vector<uchar> founded_points;
        Mat err;
        if (points_prev.size() > 0) {
            calcOpticalFlowPyrLK(mGr_prev, mGr, points_prev, points, founded_points, err,
                                 Size(25, 25), 2);

        }
        for (unsigned int i = 0; i < points_prev.size(); i++) {

            if (founded_points[i]) {
                //line(mRgb,points_prev[i],points[i],Scalar(0,255,0,255));
                //circle(mRgb,  points_prev[i], 10, Scalar(0,255,0,255));
                valid_points_prev.push_back(points_prev[i]);
                valid_points.push_back(points[i]);


            }

        }
        err.release();
    }


    Mat H;
    if (valid_points.size() > 10) {
        H = findHomography(valid_points_prev, valid_points, CV_RANSAC);
        Mat mask = Mat::zeros(mRgb.size(), CV_8U);
        // perspectiveTransform(mGr_prev,)

        Mat mGr_transformed;
        warpPerspective(mGr_prev, mGr_transformed, H, mGr_prev.size());
        //warpPerspective(mRgb_prev,mGr_transformed,H,mRgb_prev.size());

        Mat sub;
        absdiff(mGr, mGr_transformed, sub);

       // sub = sub > 80;
        Mat squ = Mat::ones(2, 2, CV_8U);
        Mat big_rect = Mat::ones(20, 60, CV_8U);


        Mat h_line = Mat::ones(4, 80, CV_8U);
        Mat v_line = Mat::ones(20, 2, CV_8U);

        erode(sub, sub, squ);
        dilate(sub, sub, Mat::ones(2, 4, CV_8UC1));



        normalize(sub, sub, 0, 255, NORM_MINMAX);
        // sub.convertTo(sub,CV_32FC1);

        sub.copyTo(output);




        sub.release();
        mGr_transformed.release();
        mask.release();
        squ.release();
        h_line.release();
        v_line.release();
    }

    H.release();
    mGr.release();
    mGr_prev.release();

    return;
}

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_FindDenseArea(JNIEnv *env,
                                                                                 jobject,
                                                                                 jlong addrMask,jintArray left,jintArray right,jintArray index) {
    Mat &input = *(Mat *) addrMask;
    Mat bin = Mat::zeros(input.size(),input.type());
    Mat mask = Mat::zeros(input.size(),input.type());

    bin = input > 32;

    input.copyTo(mask,bin);
    jint *c_left, *c_right, *c_index;

    c_left = env->GetIntArrayElements(left, 0);
    c_right = env->GetIntArrayElements(right, 0);
    c_index = env->GetIntArrayElements(index, 0);

    int width = 80;
    Rect initial_rect = cvRect(mask.cols / 2 - width/2, mask.rows / 2 - width/2, width, width);
    //cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    meanShift(mask, initial_rect, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 200, 1));
    //rectangle(mask, initial_rect.tl(), initial_rect.br(), (255, 0, 0, 255), 2, 8, 0);

    c_left[c_index[0]] = initial_rect.x + initial_rect.width/2;
    c_right[c_index[0]] = initial_rect.y + initial_rect.height/2;
    c_index[0]=c_index[0]+1;
    if(c_index[0]==c_index[1]) //circular buffer
        c_index[0]=0;

    bin.release();
    mask.release();

    env->ReleaseIntArrayElements(left, c_left, 0);
    env->ReleaseIntArrayElements(right, c_right, 0);
    env->ReleaseIntArrayElements(index, c_index, 0);


    return;


}


JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_GetClass(JNIEnv *env,
                                                                            jobject,
                                                                            jlong addInput,jstring path,jintArray class_number) {
    const char *nativeString = env->GetStringUTFChars(path, 0);
    jint *c_class_number;
    c_class_number = env->GetIntArrayElements(class_number, 0);
    string final_path = nativeString;

    Mat &input = *(Mat *) addInput;
    //CvSVM svm;

    Ptr<ml::SVM> svm = ml::SVM::create();
    BOWKMeansTrainer bowTrainer();
    const String svmfile = final_path + "/svm.xml";


    Mat dictionary;
    Mat bowDescriptor;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    Ptr<ORB> orb = ORB::create();
    BOWImgDescriptorExtractor bowDE(orb,matcher);

    FileStorage fs(final_path+"/dictionary.yml",FileStorage::READ);
    fs["vocabulary"]>>dictionary;
    fs.release();

    Mat uDictionary;
    dictionary.convertTo(uDictionary,CV_8UC1);
    bowDE.setVocabulary(uDictionary);

    //svm = ml::SVM::load<ml::SVM>(svmfile);
    svm = ml::SVM::load(svmfile);

    vector<KeyPoint> keypoint;
    orb->detect(input,keypoint);
    bowDE.compute(input,keypoint,bowDescriptor);
    float response = svm->predict(bowDescriptor);

    c_class_number[0] = (int) response;

    env->ReleaseIntArrayElements(class_number, c_class_number, 0);
    env->ReleaseStringUTFChars(path, nativeString);
}




}
