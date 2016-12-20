//
// Created by Jose Honorato on 10/12/15.
//

#include "native_processing.h"
#include <string>
using namespace std;
using namespace cv;

extern "C" {
JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_FindFeatures(JNIEnv *, jobject,
                                                                                jlong addrRgba,
                                                                                jlong addrRgba_prev) {
    Mat &mRgb_prev = *(Mat *) addrRgba_prev;
    Mat &mRgb = *(Mat *) addrRgba;

    Mat mGr, mGr_prev;
    cvtColor(mRgb, mGr, CV_BGR2GRAY);
    cvtColor(mRgb_prev, mGr_prev, CV_BGR2GRAY);


    vector<KeyPoint> kp;
    vector<KeyPoint> kp_prev;


    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mGr_prev, kp_prev);

    //Ptr<ORB> orb = ORB::create();
    //orb->setMaxFeatures(200);

    Mat desc, desc_prev;
    const float inlier_th = 2.5f;
    // orb->detectAndCompute(mGr,noArray(),kp,desc);
    //orb->detectAndCompute(mGr_prev,noArray(),kp_prev,desc_prev);


    vector<Point2f> points;
    vector<Point2f> points_prev;

    //goodFeaturesToTrack(mGr_prev,points_prev,30,0.01,30);


    for (int i = 0; i < kp_prev.size(); i++) {
        points_prev.push_back(kp_prev[i].pt);
    }

    // for(int i=0 ; i<kp_prev.size() ; i++)
    //  {
    //      points_prev.push_back(kp_prev[i].pt);
    // }


    //for (unsigned int i = 0; i <  points_prev.size(); i++) {


    // line(mRgb,points_prev[i],Point(kp2.pt.x, kp2.pt.y),Scalar(255,0,0,255));
    //  circle(mRgb,  points_prev[i], 10, Scalar(255,0,0,255));
    // }
    vector<uchar> founded_points;
    Mat err;
    if (points_prev.size() > 0) {
        calcOpticalFlowPyrLK(mGr_prev, mGr, points_prev, points, founded_points, err, Size(25, 25),
                             2);

    }

    vector<Point2f> valid_points_prev, valid_points;
    for (unsigned int i = 0; i < points_prev.size(); i++) {

        if (founded_points[i]) {
            //line(mRgb,points_prev[i],points[i],Scalar(0,255,0,255));
            // circle(mRgb,  points_prev[i], 10, Scalar(0,255,0,255));
            valid_points_prev.push_back(points_prev[i]);
            valid_points.push_back(points[i]);


        }

    }
    Mat H;
    if (valid_points.size() > 10) {
        H = findHomography(valid_points_prev, valid_points, CV_RANSAC);
        Mat mask = Mat::zeros(mRgb.size(), CV_8U);
        for (int i = 0; i < valid_points.size(); i++) {
            Mat v1 = Mat::ones(3, 1, CV_64F);
            v1.at<double>(0) = valid_points_prev[i].x;
            v1.at<double>(1) = valid_points_prev[i].y;
//
            v1 = H * v1;
            v1 /= v1.at<double>(2);
//
            double error = sqrt(pow(v1.at<double>(0) - valid_points[i].x, 2) +
                                pow(v1.at<double>(1) - valid_points[i].y, 2));
//
            if (error > inlier_th) {
                mask.at<uchar>(valid_points[i].y, valid_points[i].x) = 1;
                //line(mRgb,valid_points_prev[i],valid_points[i],Scalar(255,0,0,255));
                //circle(mRgb,  valid_points_prev[i], 10, Scalar(255,0,0,255));

            }
//
//
            v1.release();
        }
        Mat squ = Mat::ones(5, 5, CV_8U);
        Mat h_line = Mat::ones(5, 200, CV_8U);
        Mat v_line = Mat::ones(100, 5, CV_8U);

        dilate(mask, mask, squ);

        dilate(mask, mask, h_line);
        erode(mask, mask, h_line);

        dilate(mask, mask, v_line);
        erode(mask, mask, v_line);
        vector<Vec4i> hierarchy;
        vector<vector<Point> > contours;
        findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        for (int i = 0; i < contours.size(); i++) {
            vector<Point> contour_poly;
            approxPolyDP(Mat(contours[i]), contour_poly, 3, true);
            Rect bound_rect = boundingRect(Mat(contour_poly));
            Mat mat_rect = mRgb(bound_rect);

            rectangle(mRgb, bound_rect.tl(), bound_rect.br(), (255, 0, 0, 255), 2, 8, 0);
        }

        //Mat temp= Mat::zeros(mRgb.size(),mRgb.type());
        //mRgb.copyTo(temp,mask);
        // mRgb = Mat::ones(temp.size(),temp.type());
        // temp.copyTo(mRgb);
    }

    H.release();







//    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
//
//    vector< vector<DMatch> > matches;
//    const double nn_match_ratio = 0.8f;
//    vector<KeyPoint> matched1, matched2;
//
//    matcher->knnMatch(desc_prev, desc, matches, 2);
//
//    for(unsigned i = 0; i < matches.size(); i++) {
//        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
//            matched1.push_back(kp_prev[matches[i][0].queryIdx]);
//            matched2.push_back(      kp[matches[i][0].trainIdx]);
//        }
//    }
//
//    for (unsigned int i = 0; i < matched2.size(); i++) {
//        const KeyPoint& kp1 = matched1[i];
//        const KeyPoint& kp2 = matched2[i];
//
//        line(mRgb,Point(kp1.pt.x, kp1.pt.y),Point(kp2.pt.x, kp2.pt.y),Scalar(255,0,0,255));
//        circle(mRgb, Point(kp2.pt.x, kp2.pt.y), 10, Scalar(255,0,0,255));
//    }
}

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_IncreaseContrast(JNIEnv *,
                                                                                    jobject,
                                                                                    jlong addrRgba) {
    Mat &src = *(Mat *) addrRgba;
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
std::string ConvertJString(JNIEnv *env, jstring str) {
    if (!str) String();

    const jsize len = env->GetStringUTFLength(str);
    const char *strChars = env->GetStringUTFChars(str, (jboolean *) 0);

    std::string Result(strChars, len);

    env->ReleaseStringUTFChars(str, strChars);

    return Result;
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

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_SaveBoundles(JNIEnv *env,
                                                                                jobject,
                                                                                jlong addrRgba,
                                                                                jlong addrRgba_prev,
                                                                                jstring path) {
    const char *nativeString = env->GetStringUTFChars(path, 0);
    string final_path = nativeString;

    Mat &mRgb_prev = *(Mat *) addrRgba_prev;
    Mat &mRgb = *(Mat *) addrRgba;

    Mat mGr, mGr_prev;
    cvtColor(mRgb, mGr, CV_BGR2GRAY);
    cvtColor(mRgb_prev, mGr_prev, CV_BGR2GRAY);

    vector<KeyPoint> kp;
    vector<KeyPoint> kp_prev;


    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mGr_prev, kp_prev);
    /////////////////////////////////////
    detector->detect(mGr, kp);
    //////////////////////////////////
    vector<Point2f> valid_points_prev, valid_points;
    bool optical_flow = true;
    const float inlier_th = 2.5f;

    if (optical_flow == true) {

        Mat desc, desc_prev;

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
                line(mRgb, points_prev[i], points[i], Scalar(0, 255, 0, 255));
                circle(mRgb, points_prev[i], 10, Scalar(0, 255, 0, 255));
                valid_points_prev.push_back(points_prev[i]);
                valid_points.push_back(points[i]);


            }

        }
    }
    else {
        Ptr<ORB> orb = ORB::create();
        orb->setMaxFeatures(200);

        Mat desc, desc_prev;
        orb->compute(mGr, kp, desc);
        orb->compute(mGr_prev, kp_prev, desc_prev);

        ///////////////////////////////////
        BFMatcher bfmatcher(NORM_HAMMING, true);
        vector<DMatch> matches;
        bfmatcher.match(desc_prev, desc, matches);
        KeyPoint kp1, kp2;
        float min_dist = 100;
        for (int i = 0; i < matches.size(); i++) {
            if (matches[i].distance < min_dist) {
                min_dist = matches[i].distance;
            }

        }

        for (int i = 0; i < matches.size(); i++) {
            if (matches[i].distance <= max(20 * min_dist, (float) 0.02)) {
                kp1 = kp_prev[matches[i].queryIdx];
                kp2 = kp[matches[i].trainIdx];

                valid_points_prev.push_back(kp1.pt);
                valid_points.push_back(kp2.pt);
                line(mRgb, kp1.pt, kp2.pt, Scalar(0, 0, 255, 255));
                circle(mRgb, kp1.pt, 10, Scalar(0, 0, 255, 255));
            }

        }

    }

    Mat H;
    if (valid_points.size() > 10) {
        H = findHomography(valid_points_prev, valid_points, CV_RANSAC);
        Mat mask = Mat::zeros(mRgb.size(), CV_8U);
        for (int i = 0; i < valid_points.size(); i++) {
            Mat v1 = Mat::ones(3, 1, CV_64F);
            v1.at<double>(0) = valid_points_prev[i].x;
            v1.at<double>(1) = valid_points_prev[i].y;
//
            v1 = H * v1;
            v1 /= v1.at<double>(2);
//
            double error = sqrt(pow(v1.at<double>(0) - valid_points[i].x, 2) +
                                pow(v1.at<double>(1) - valid_points[i].y, 2));
//
            if (error > inlier_th) {
                mask.at<uchar>(valid_points[i].y, valid_points[i].x) = 1;
                line(mRgb, valid_points_prev[i], valid_points[i], Scalar(255, 0, 0, 255));
                circle(mRgb, valid_points_prev[i], 10, Scalar(255, 0, 0, 255));

            }
//
//
            v1.release();
        }
        Mat squ = Mat::ones(5, 5, CV_8U);
        Mat h_line = Mat::ones(5, 200, CV_8U);
        Mat v_line = Mat::ones(100, 5, CV_8U);

        dilate(mask, mask, squ);

        dilate(mask, mask, h_line);
        erode(mask, mask, h_line);

        dilate(mask, mask, v_line);
        erode(mask, mask, v_line);
        vector<Vec4i> hierarchy;
        vector<vector<Point> > contours;
        findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        for (int i = 0; i < contours.size(); i++) {
            vector<Point> contour_poly;
            approxPolyDP(Mat(contours[i]), contour_poly, 3, true);
            Rect bound_rect = boundingRect(Mat(contour_poly));
            Mat mat_rect = mRgb(bound_rect);
            char name[30];
            sprintf(name, "%d.png", i);
            imwrite(final_path + "/" + name, mat_rect);


            rectangle(mRgb, bound_rect.tl(), bound_rect.br(), (255, 0, 0, 255), 2, 8, 0);
        }
        mask.release();
        squ.release();
        h_line.release();
        v_line.release();

    }

    H.release();
    env->ReleaseStringUTFChars(path, nativeString);
    return;

}

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_GetBiggestBoundle(JNIEnv *env,
                                                                                     jobject,
                                                                                     jlong addrRgba,
                                                                                     jlong addrRgba_prev,
                                                                                     jintArray rect_data) {

    Mat &mRgb_prev = *(Mat *) addrRgba_prev;
    Mat &mRgb = *(Mat *) addrRgba;

    jint *c_rect_data;
    c_rect_data = env->GetIntArrayElements(rect_data, 0);

    Mat mGr, mGr_prev;
    cvtColor(mRgb, mGr, CV_BGR2GRAY);
    cvtColor(mRgb_prev, mGr_prev, CV_BGR2GRAY);

    vector<KeyPoint> kp;
    vector<KeyPoint> kp_prev;


    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mGr_prev, kp_prev);
    /////////////////////////////////////
    detector->detect(mGr, kp);
    //////////////////////////////////
    vector<Point2f> valid_points_prev, valid_points;
    bool optical_flow = true;
    const float inlier_th = 2.5f;

    Mat desc, desc_prev;

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
            line(mRgb, points_prev[i], points[i], Scalar(0, 255, 0, 255));
            circle(mRgb, points_prev[i], 10, Scalar(0, 255, 0, 255));
            valid_points_prev.push_back(points_prev[i]);
            valid_points.push_back(points[i]);


        }

    }

    Mat H;
    if (valid_points.size() > 10) {
        H = findHomography(valid_points_prev, valid_points, CV_RANSAC);
        Mat mask = Mat::zeros(mRgb.size(), CV_8U);
        for (int i = 0; i < valid_points.size(); i++) {
            Mat v1 = Mat::ones(3, 1, CV_64F);
            v1.at<double>(0) = valid_points_prev[i].x;
            v1.at<double>(1) = valid_points_prev[i].y;

            v1 = H * v1;
            v1 /= v1.at<double>(2);

            double error = sqrt(pow(v1.at<double>(0) - valid_points[i].x, 2) +
                                pow(v1.at<double>(1) - valid_points[i].y, 2));

            if (error > inlier_th) {
                mask.at<uchar>(valid_points[i].y, valid_points[i].x) = 1;
                line(mRgb, valid_points_prev[i], valid_points[i], Scalar(255, 0, 0, 255));
                circle(mRgb, valid_points_prev[i], 10, Scalar(255, 0, 0, 255));

            }
            v1.release();
        }

        Mat squ = Mat::ones(10, 10, CV_8U);
        Mat h_line = Mat::ones(5, 200, CV_8U);
        Mat v_line = Mat::ones(100, 5, CV_8U);

        dilate(mask, mask, squ);
        erode(mask, mask, squ);


        dilate(mask, mask, h_line);
        erode(mask, mask, h_line);

        dilate(mask, mask, v_line);
        erode(mask, mask, v_line);
        vector<Vec4i> hierarchy;
        vector<vector<Point> > contours;
        findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        float max_area = 0;
        int index = -1;
        Rect biggest_rect;
        for (int i = 0; i < contours.size(); i++) {
            vector<Point> contour_poly;
            approxPolyDP(Mat(contours[i]), contour_poly, 3, true);
            Rect bound_rect = boundingRect(Mat(contour_poly));
            rectangle(mRgb, bound_rect.tl(), bound_rect.br(), (255, 0, 0, 255), 2, 8, 0);
            float area = contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                index = i;
                biggest_rect = bound_rect;
            }

        }

        if (max_area > 50) {
            c_rect_data[0] = biggest_rect.x;
            c_rect_data[1] = biggest_rect.y;
            c_rect_data[2] = biggest_rect.width;
            c_rect_data[3] = biggest_rect.height;


        }
        else//nothing to track
            c_rect_data[0] = -1;

        //release memory
        mask.release();
        squ.release();
        h_line.release();
        v_line.release();

    }

    H.release();
    env->ReleaseIntArrayElements(rect_data, c_rect_data, 0);

    return;

}
JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_TrackRect(JNIEnv *env, jobject,
                                                                             jlong addrRgba,
                                                                             jlong addrRgba_prev,
                                                                             jintArray rect_data) {

    Mat &mRgb_prev = *(Mat *) addrRgba_prev;
    Mat &mRgb = *(Mat *) addrRgba;

    int motion_th = 8;
    jint *c_rect_data;
    c_rect_data = env->GetIntArrayElements(rect_data, 0);


    Mat mGr, mGr_prev;
    cvtColor(mRgb, mGr, CV_BGR2GRAY);
    cvtColor(mRgb_prev, mGr_prev, CV_BGR2GRAY);

    Rect tracking_bound(c_rect_data[0], c_rect_data[1], c_rect_data[2], c_rect_data[3]);
    c_rect_data[0] = -1;


//    int increase_bound = 50;
//    int new_x = c_rect_data[0]-increase_bound/2;
//    int new_y = c_rect_data[1]-increase_bound/2;
//    int new_w = increase_bound;
//    int new_h = increase_bound;
//
//    if(new_x<0)
//        new_x = 0;
//    if(new_y<0)
//        new_y = 0;
//    if(new_w+new_x>mGr.cols)
//        new_w = mGr.cols-1 - new_x;
//    if(new_h+new_y>mGr.rows)
//        new_h = mGr.rows-1 - new_y;
//
//    Rect search_rect(new_x,new_y,new_w,new_h);
//    rectangle(mRgb,search_rect.tl(),search_rect.br(),(255,0,0,255),2,8,0);
    Mat tracking_mat_prev = mGr_prev(tracking_bound);
    //  Mat search_mat = mGr(search_rect);


    vector<KeyPoint> kp;
    vector<KeyPoint> kp_prev;


    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(tracking_mat_prev, kp_prev);
    /////////////////////////////////////
    //  detector->detect(search_mat,kp);
    //////////////////////////////////
    vector<Point2f> valid_points_prev, valid_points;
    bool optical_flow = true;
    const float inlier_th = 2.5f;

    Mat desc, desc_prev;

    vector<Point2f> points;
    vector<Point2f> points_prev;

    for (int i = 0; i < kp_prev.size(); i++) {
        // points_prev.push_back(Point2f(kp_prev[i].pt.x,kp_prev[i].pt.y));
        points_prev.push_back(
                Point2f(kp_prev[i].pt.x + tracking_bound.x, kp_prev[i].pt.y + tracking_bound.y));
    }
    //  Mat temp_prev = Mat::zeros(search_mat.size(),search_mat.type());
    // temp_prev(tracking_bound)=tracking_mat_prev;


    if (points_prev.size() > 0) {
        vector<uchar> founded_points;
        Mat err;
        //calcOpticalFlowPyrLK(tracking_mat_prev, search_mat, points_prev, points, founded_points, err,Size(25, 25), 2);
        calcOpticalFlowPyrLK(mGr_prev, mGr, points_prev, points, founded_points, err,
                             Size(25, 25), 2);


        Mat mask = Mat::zeros(mRgb.size(), CV_8U);
        float sum_x = 0, sum_y = 0;
        int count_valids = 0;
        for (unsigned int i = 0; i < founded_points.size(); i++) {

            if (founded_points[i]) {
                //mask.at<uchar>(points[i].y+search_rect.y,points[i].x+search_rect.x) = 1;

                int x_motion = points[i].x - points_prev[i].x;
                int y_motion = points[i].y - points_prev[i].y;

                if (x_motion > motion_th || y_motion > motion_th) {
                    sum_x += (x_motion);
                    sum_y += (y_motion);
                    count_valids++;
                }


                mask.at<uchar>(points[i].y, points[i].x) = 1;
                line(mRgb, points_prev[i], points[i], Scalar(0, 255, 0, 255));
                circle(mRgb, points_prev[i], 10, Scalar(0, 255, 0, 255));


            }

        }

        if (count_valids > 10) {
            int new_trackRect_x = tracking_bound.x + (int) (sum_x / count_valids);
            int new_trackRect_y = tracking_bound.y + (int) (sum_y / count_valids);
            c_rect_data[0] = new_trackRect_x;
            c_rect_data[1] = new_trackRect_y;
            c_rect_data[2] = c_rect_data[2];
            c_rect_data[3] = c_rect_data[3];

        }
        else
            c_rect_data[0] = -1;//no rect
        mask.release();
        err.release();
    }


    desc.release();
    desc_prev.release();
    //search_mat.release();
    tracking_mat_prev.release();
    mGr.release();
    mGr_prev.release();

    env->ReleaseIntArrayElements(rect_data, c_rect_data, 0);

    return;


}

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_FindInRect(JNIEnv *env, jobject,
                                                                              jlong addrRgba,
                                                                              jlong addrRgba_prev,
                                                                              jintArray rect_data) {

    Mat &mRgb_prev = *(Mat *) addrRgba_prev;
    Mat &mRgb = *(Mat *) addrRgba;

    int motion_th = 8;
    jint *c_rect_data;
    c_rect_data = env->GetIntArrayElements(rect_data, 0);


    Mat mGr, mGr_prev;
    cvtColor(mRgb, mGr, CV_BGR2GRAY);
    cvtColor(mRgb_prev, mGr_prev, CV_BGR2GRAY);

    Rect tracking_bound(c_rect_data[0], c_rect_data[1], c_rect_data[2], c_rect_data[3]);
    c_rect_data[0] = -1;


//    int increase_bound = 50;
//    int new_x = c_rect_data[0]-increase_bound/2;
//    int new_y = c_rect_data[1]-increase_bound/2;
//    int new_w = increase_bound;
//    int new_h = increase_bound;
//
//    if(new_x<0)
//        new_x = 0;
//    if(new_y<0)
//        new_y = 0;
//    if(new_w+new_x>mGr.cols)
//        new_w = mGr.cols-1 - new_x;
//    if(new_h+new_y>mGr.rows)
//        new_h = mGr.rows-1 - new_y;
//
//    Rect search_rect(new_x,new_y,new_w,new_h);
//    rectangle(mRgb,search_rect.tl(),search_rect.br(),(255,0,0,255),2,8,0);
    Mat tracking_mat_prev = mGr_prev(tracking_bound);
    //  Mat search_mat = mGr(search_rect);


    vector<KeyPoint> kp;
    vector<KeyPoint> kp_prev;


    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(tracking_mat_prev, kp_prev);
    /////////////////////////////////////
    //  detector->detect(search_mat,kp);
    //////////////////////////////////
    vector<Point2f> valid_points_prev, valid_points;
    bool optical_flow = true;
    const float inlier_th = 2.5f;

    Mat desc, desc_prev;

    vector<Point2f> points;
    vector<Point2f> points_prev;

    for (int i = 0; i < kp_prev.size(); i++) {
        // points_prev.push_back(Point2f(kp_prev[i].pt.x,kp_prev[i].pt.y));
        points_prev.push_back(
                Point2f(kp_prev[i].pt.x + tracking_bound.x, kp_prev[i].pt.y + tracking_bound.y));
    }
    //  Mat temp_prev = Mat::zeros(search_mat.size(),search_mat.type());
    // temp_prev(tracking_bound)=tracking_mat_prev;


    if (points_prev.size() > 0) {
        vector<uchar> founded_points;
        Mat err;
        //calcOpticalFlowPyrLK(tracking_mat_prev, search_mat, points_prev, points, founded_points, err,Size(25, 25), 2);
        calcOpticalFlowPyrLK(mGr_prev, mGr, points_prev, points, founded_points, err,
                             Size(25, 25), 2);

        for (unsigned int i = 0; i < points_prev.size(); i++) {

            if (founded_points[i]) {
                line(mRgb, points_prev[i], points[i], Scalar(0, 255, 0, 255));
                circle(mRgb, points_prev[i], 10, Scalar(0, 255, 0, 255));
                valid_points_prev.push_back(points_prev[i]);
                valid_points.push_back(points[i]);


            }

        }

        Mat H;
        if (valid_points.size() > 10) {
            H = findHomography(valid_points_prev, valid_points, CV_RANSAC);
            Mat mask = Mat::zeros(mRgb.size(), CV_8U);
            float sum_x = 0, sum_y = 0;
            int count_valids = 0;
            for (int i = 0; i < valid_points.size(); i++) {
                Mat v1 = Mat::ones(3, 1, CV_64F);
                v1.at<double>(0) = valid_points_prev[i].x;
                v1.at<double>(1) = valid_points_prev[i].y;

                v1 = H * v1;
                v1 /= v1.at<double>(2);

                double error = sqrt(pow(v1.at<double>(0) - valid_points[i].x, 2) +
                                    pow(v1.at<double>(1) - valid_points[i].y, 2));

                if (error <
                    inlier_th)//this time we want to find inliers which refer to the object movement because the mejority of points are from the moving object
                {
                    sum_x += (valid_points[i].x - valid_points_prev[i].x);
                    sum_y += (valid_points[i].y - valid_points_prev[i].y);
                    count_valids++;

                    mask.at<uchar>(valid_points[i].y, valid_points[i].x) = 1;
                    line(mRgb, valid_points_prev[i], valid_points[i], Scalar(255, 0, 0, 255));
                    circle(mRgb, valid_points_prev[i], 10, Scalar(255, 0, 0, 255));

                }
                v1.release();
            }

            if (count_valids > 20) {
                c_rect_data[0] = tracking_bound.x + sum_x / count_valids;
                c_rect_data[1] = tracking_bound.y + sum_y / count_valids;
                c_rect_data[2] = tracking_bound.width;
                c_rect_data[3] = tracking_bound.height;


            }
            else//nothing to track
                c_rect_data[0] = -1;


//            Mat squ = Mat::ones(10,10,CV_8U);
//            Mat h_line = Mat::ones(5,200,CV_8U);
//            Mat v_line = Mat::ones(100,5,CV_8U);
//
//            dilate(mask,mask,squ);
//            erode(mask,mask,squ);
//
//
//            dilate(mask,mask,h_line);
//            erode(mask,mask,h_line);
//
//            dilate(mask,mask,v_line);
//            erode(mask,mask,v_line);
//            vector<Vec4i> hierarchy;
//            vector<vector<Point> > contours;
//            findContours(mask,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
//
//            float  max_area=0;
//            int index=-1;
//            Rect biggest_rect;
//            for(int i=0; i<contours.size();i++)
//            {
//                vector<Point> contour_poly;
//                approxPolyDP(Mat(contours[i]),contour_poly,3,true);
//                Rect bound_rect = boundingRect(Mat(contour_poly));
//                rectangle(mRgb,bound_rect.tl(),bound_rect.br(),(255,0,0,255),2,8,0);
//                float area = contourArea(contours[i]);
//                if(area>max_area)
//                {
//                    max_area = area;
//                    index = i;
//                    biggest_rect = bound_rect;
//                }
//
//            }
//
//            if(max_area>50)
//            {
//                c_rect_data[0] = biggest_rect.x;
//                c_rect_data[1] = biggest_rect.y;
//                c_rect_data[2] = biggest_rect.width;
//                c_rect_data[3] = biggest_rect.height;
//
//
//            }
//            else//nothing to track
//                c_rect_data[0] = -1;

            //release memory
            mask.release();
            //      squ.release();
            //      h_line.release();
            //      v_line.release();
            H.release();

        }


    }


    desc.release();
    desc_prev.release();
    //search_mat.release();
    tracking_mat_prev.release();
    mGr.release();
    mGr_prev.release();

    env->ReleaseIntArrayElements(rect_data, c_rect_data, 0);

    return;
}

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_MatchRect(JNIEnv *env, jobject,
                                                                             jlong addrRgba,
                                                                             jlong addrRgba_prev,
                                                                             jintArray rect_data,
                                                                             jdoubleArray value_data) {

    Mat &mRgb_prev = *(Mat *) addrRgba_prev;
    Mat &mRgb = *(Mat *) addrRgba;

    int motion_th = 8;
    jint *c_rect_data;
    jdouble *c_value_data;

    c_rect_data = env->GetIntArrayElements(rect_data, 0);
    c_value_data = env->GetDoubleArrayElements(value_data, 0);


    Mat mGr, mGr_prev;
    cvtColor(mRgb, mGr, CV_BGR2GRAY);
    cvtColor(mRgb_prev, mGr_prev, CV_BGR2GRAY);

    Rect tracking_bound(c_rect_data[0], c_rect_data[1], c_rect_data[2], c_rect_data[3]);
    c_rect_data[0] = -1;


//    int increase_bound = 50;
//    int new_x = c_rect_data[0]-increase_bound/2;
//    int new_y = c_rect_data[1]-increase_bound/2;
//    int new_w = increase_bound;
//    int new_h = increase_bound;
//
//    if(new_x<0)
//        new_x = 0;
//    if(new_y<0)
//        new_y = 0;
//    if(new_w+new_x>mGr.cols)
//        new_w = mGr.cols-1 - new_x;
//    if(new_h+new_y>mGr.rows)
//        new_h = mGr.rows-1 - new_y;
//
//    Rect search_rect(new_x,new_y,new_w,new_h);
//    rectangle(mRgb,search_rect.tl(),search_rect.br(),(255,0,0,255),2,8,0);
    Mat tracking_mat_prev = mGr_prev(tracking_bound);
    //  Mat search_mat = mGr(search_rect);

    int result_cols = mGr.cols - tracking_mat_prev.cols + 1;
    int result_rows = mGr.rows - tracking_mat_prev.rows + 1;

    Mat result(result_rows, result_cols, CV_32FC1);

    matchTemplate(mGr, tracking_mat_prev, result, CV_TM_SQDIFF);

    //normalize(result,result,0,1,NORM_MINMAX,-1,Mat());

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    Point matchLoc = minLoc;//for SQDIFF the minimum value is the match


    // if(count_valids>20)
    //should set some if for false matching (without answer cases)
    if (minVal < 100000000)//threshold which sures we have true match
    {
        c_rect_data[0] = matchLoc.x;
        c_rect_data[1] = matchLoc.y;
        c_rect_data[2] = tracking_bound.width;
        c_rect_data[3] = tracking_bound.height;
        c_value_data[0] = minVal;
    }
    else//nothing to track
        c_rect_data[0] = -1;










    //search_mat.release();
    tracking_mat_prev.release();
    mGr.release();
    mGr_prev.release();

    env->ReleaseIntArrayElements(rect_data, c_rect_data, 0);
    env->ReleaseDoubleArrayElements(value_data, c_value_data, 0);


    return;
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

JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_HasCar(JNIEnv *env,
                                                                                 jobject,
                                                                                 jlong addInput,jintArray car_number) {

    Mat &input = *(Mat *) addInput;
    jint *c_car_number;
    c_car_number = env->GetIntArrayElements(car_number, 0);

    ////Nothing




    env->ReleaseIntArrayElements(car_number, c_car_number, 0);





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

    svm = ml::SVM::load<ml::SVM>(svmfile);

    vector<KeyPoint> keypoint;
    orb->detect(input,keypoint);
    bowDE.compute(input,keypoint,bowDescriptor);
    float response = svm->predict(bowDescriptor);

    c_class_number[0] = (int) response;

    env->ReleaseIntArrayElements(class_number, c_class_number, 0);
    env->ReleaseStringUTFChars(path, nativeString);
}


JNIEXPORT void JNICALL Java_org_honorato_opencvsample_MainActivity_testInt(JNIEnv *env, jobject,
                                                                           jlong addrRgba,
                                                                           jintArray input) {


    jint *body = (env)->GetIntArrayElements(input, 0);
    body[0] = 500;
    (env)->ReleaseIntArrayElements(input, body, 0);

    //input[0] = 500;
    return;
}


}


