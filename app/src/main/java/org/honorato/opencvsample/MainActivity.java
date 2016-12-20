package org.honorato.opencvsample;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.*;



import org.opencv.videoio.VideoCapture;

import java.io.File;
import static java.lang.Math.hypot;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC1;


public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
   // Mat a = Mat.ones(10,10);



    CascadeClassifier carDetector;
    File basic_path = new File(Environment.getExternalStorageDirectory()+"/myImages");
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        System.loadLibrary("native");
        //final Button button = (Button) findViewById(R.id.button);

        // Example of a call to a native method
        TextView tv = (TextView) findViewById(R.id.TV_1);
       //
        basic_path.mkdir();
        String filename = basic_path.toString()+"/"+Integer.toString(1)+".png";
        String video_path = basic_path.toString()+"/"+"vid1.mp4";

       // Mat input = Imgcodecs.imread(filename);
       // VideoCapture cap = new VideoCapture(video_path);

//        if(cap.isOpened())
//        {
//            cap.read(input);
//            cap.release();
//        }
//        else
//            showToast("videoError: couldn't open "+video_path);
//        filename = basic_path.toString()+"/"+Integer.toString(1)+"_cnv.png";
//        String dd="salam";
//        String temp = stringFromJNI(dd);
//        IncreaseContrast(input.getNativeObjAddr());
//
//        boolean result = Imgcodecs.imwrite(filename,input);
//        if(result)
//            showToast(filename);
//        else
//            showToast("error");
        tv.setText("salam");

    }

    public void PB2_click(View view)
    {
        int[] nums=new int[3];

        String temp = stringFromJNI("dd",nums);
        TextView tv = (TextView) findViewById(R.id.TV_3);
        tv.setText(String.format("%d",nums[0]));


    }

    public void PB_track_click(View view)
    {
        showToast("salam");
        TextView tv = (TextView) findViewById(R.id.TV_1);
        EditText ET_start = (EditText) findViewById(R.id.ET_start);
        EditText ET_end = (EditText) findViewById(R.id.ET_end);

        tv.setText("start");
        String images_folder = basic_path+"/KITTI_1";
        String out_folder = basic_path+"/KITTI_1/output9";
        File out_file = new File(out_folder);
        out_file.mkdir();

        Mat input;
        showToast("start");
        // long startTime = System.nanoTime();
        Mat prev=Mat.eye(2,2,CV_8U);
        int start_frame = (int)Long.parseLong(ET_start.getText().toString());
        int end_frame =(int) Long.parseLong(ET_end.getText().toString());

        boolean tracking = false;
        Rect tracking_bound = new Rect(0,0,3,3);
        String min_values = "";


        int[] rect_data = new int[5];
        for(int i=start_frame;i<end_frame;i++)
        {
            //input = Imgcodecs.imread(images_folder+"/image_"+String.format("%08d_0", i)+".png");
            input = Imgcodecs.imread(images_folder+"/"+String.format("%010d", i)+".png");

            if(i==start_frame)
            {
                prev = Mat.zeros(input.size(),input.type());
                input.copyTo(prev);
            }

            else
            {
                if(tracking==false)
                {
                    Mat temp = Mat.zeros(input.size(), input.type());
                    input.copyTo(temp);

                    rect_data[0] = -1;//default not found
                    GetBiggestBoundle(input.getNativeObjAddr(), prev.getNativeObjAddr(), rect_data);
                    boolean result = Imgcodecs.imwrite(out_folder + "/image_" + String.format("%08d", i) + "_cnv.png", input);
                    temp.copyTo(prev);
                    temp.release();

                    if (rect_data[0] != -1) {
                        tracking = true;
                        tracking_bound = new Rect(rect_data[0], rect_data[1], rect_data[2], rect_data[3]);
                    }
                }
                else
                {
                    Mat temp = Mat.zeros(input.size(), input.type());
                    input.copyTo(temp);
                    double [] value_data = new double [1];
                    MatchRect(input.getNativeObjAddr(), prev.getNativeObjAddr(), rect_data,value_data);
                    if (rect_data[0] != -1) {
                        tracking = true;
                        tracking_bound = new Rect(rect_data[0], rect_data[1], rect_data[2], rect_data[3]);
                        Imgproc.rectangle(input,tracking_bound.tl(),tracking_bound.br(),new Scalar(255,0,0,255),2,8,0);
                        min_values=min_values+"_"+String.format("%f",value_data[0]);
                    }
                    else
                        tracking = false;
                    boolean result = Imgcodecs.imwrite(out_folder + "/image_" + String.format("%08d", i) + "_cnv.png", input);
                    temp.copyTo(prev);
                    temp.release();

                }

            }

        }
        tv.setText(String.format("end:%d_values:",start_frame)+min_values);

        showToast("end");

    }
    public void PB_transform_click(View view)
    {
        showToast("salam");
        TextView tv = (TextView) findViewById(R.id.TV_1);
        EditText ET_start = (EditText) findViewById(R.id.ET_start);
        EditText ET_end = (EditText) findViewById(R.id.ET_end);

        tv.setText("start");
        String images_folder = basic_path+"/HKUST_2";
        String out_folder = basic_path+"/HKUST_2/HKUST_2_result";
        File out_file = new File(out_folder);
        out_file.mkdir();

        Mat input;
        showToast("start");
        // long startTime = System.nanoTime();
        Mat prev=Mat.eye(2,2,CV_8U);
        int start_frame = (int)Long.parseLong(ET_start.getText().toString());
        int end_frame =(int) Long.parseLong(ET_end.getText().toString());

        boolean tracking = false;
        Rect tracking_bound = new Rect(0,0,3,3);
        String min_values = "";


        int[] rect_data = new int[5];
        int[] left_point = new int[10];
        int[] right_point = new int[10];
        int[] index = new int[2];

        int[] class_number = new int[1];

        index[1] = 10; //size of circular buffer



        int resize_cols,resize_rows;

        input = Imgcodecs.imread(images_folder+"/images/"+String.format("%010d", start_frame)+".png");
        resize_cols = input.cols()/4;
        resize_rows = input.rows()/4;
        Mat output = Mat.zeros(resize_rows,resize_cols,CV_8UC1);
        Mat normal = Mat.zeros(resize_rows,resize_cols,CV_8UC1);

        int counter = 0;

        long sum_mask=0, sum_class=0 , sum_all=0;
        long mask_counter =0, class_counter =0, all_counter=0;


        for (int i = start_frame + 1; i < end_frame; i++) {
            input = Imgcodecs.imread(images_folder + "/images/" + String.format("%010d", i) + ".png");
            resize_cols = input.cols() / 4;
            resize_rows = input.rows() / 4;
            Mat input_resized = Mat.zeros(resize_rows, resize_cols, input.type());

            Size size2 = new Size(resize_cols, resize_rows);
            Imgproc.resize(input, input_resized, size2);



            if (i == start_frame + 1) {
                prev = Mat.zeros(input_resized.size(), input_resized.type());
                input_resized.copyTo(prev);
            } else {
                counter++;
                long MaskStartTime = System.nanoTime();
                long AllStartTime = System.nanoTime();
                Mat temp = Mat.zeros(input_resized.size(), input_resized.type());
                input_resized.copyTo(temp);
                //output = Mat.zeros(resize_rows,resize_cols,CV_32FC1);
                CreateMask(input_resized.getNativeObjAddr(), prev.getNativeObjAddr(), output.getNativeObjAddr());
                //output.convertTo(output,CV_32FC1);
                Core.add(output, normal, normal);
                Imgproc.dilate(normal, normal, Mat.ones(2, 8, CV_8UC1));


                Core.normalize(normal, normal, 0, 126, Core.NORM_MINMAX);
                Mat tosave = Mat.zeros(normal.size(), CV_8UC1);
                normal.copyTo(tosave);
                FindDenseArea(tosave.getNativeObjAddr(), left_point, right_point, index);
                if (counter > index[1])//buffer has been filled
                {
                    draw_points(tosave, left_point, right_point, index);
                    double mean = get_mean(left_point, right_point, index);
                    long MaskEndTime = System.nanoTime();
                    long MaskTime = MaskEndTime - MaskStartTime;
                    sum_mask = sum_mask + MaskTime/1000000;
                    mask_counter= mask_counter + 1;

                    //Double.toString(mean)
                    //  Imgproc.putText(tosave,Double.toString(mean),new Point(50,50),3,(double) 2,new Scalar(255,0,0,255),3);
                    if (mean < 7) {
                        long ClassStartTime = System.nanoTime();
                        Rect bound = get_proposal_rect(input_resized, left_point, right_point, index);
                        Mat mat_bound = new Mat(input,new Rect(bound.x*4,bound.y*4,bound.width*4,bound.height*4));
                        GetClass(mat_bound.getNativeObjAddr(),basic_path.toString(),class_number);
                        Imgproc.rectangle(input_resized, bound.tl(), bound.br(), new Scalar(255, 0, 0, 255), 2, 8, 0);
                        Imgproc.putText(input_resized,String.format("%dc:%d",(int)mean,class_number[0]),bound.tl(),3,(double) 1,new Scalar(255,0,0,255),3);
                        long ClassEndTime = System.nanoTime();
                        long ClassTime = ClassEndTime - ClassStartTime;

                        sum_class = sum_class + ClassTime/1000000;
                        class_counter= class_counter + 1;

                    }

                }

                long AllEndTime = System.nanoTime();
                long AllTime = AllEndTime - AllStartTime;
                sum_all = sum_all + AllTime/1000000;
                all_counter= all_counter + 1;

                boolean result = Imgcodecs.imwrite(out_folder + "/image_" + String.format("%08d", i) + "_cnv.png", input_resized);



                temp.copyTo(prev);
                temp.release();
                output.release();
                tosave.release();
                input_resized.release();
                tosave.release();
            }

            input_resized.release();


            tv.setText(String.format("mask:%f class:%f all:%f",(float)sum_mask/mask_counter,(float)sum_class/class_counter,(float)sum_all/all_counter));

            showToast("end");
        }

    }




    public Rect get_proposal_rect(Mat input,int [] left,int[] right, int index[])
    {
        int last = index[0]-1;
        int size = index[1];

        if(last==-1)
            last = size-1;

        int center_x = left[last];
        int center_y = right[last];

        int width = 80; // same as meanshift rect

        int rect_x = center_x - width/2;
        int rect_y = center_y - width/2;

        if(rect_x < 0)
            rect_x = 0;
        if(rect_y < 0)
            rect_y = 0;

        if(rect_x + width > input.cols()-1)
            width = input.cols() -1 - rect_x;

        if(rect_y + width > input.rows()-1)
            width = input.rows() -1 - rect_y;
        Rect bounding = new Rect(rect_x,rect_y,width,width);
        return bounding;
       // Imgproc.rectangle(input,bounding.tl(),bounding.br(),new Scalar(255,0,0,255),2,8,0);


    }
    public void draw_points(Mat input,int [] left,int[] right, int index[])
    {
        int first_ind = index[0];
        int second_ind = index[0] + 1;
        int size = index[1];

        if(second_ind==size) {
            second_ind = 0;
        }

        for(int i=0;i<index[1]-1;i++)
        {
            Imgproc.line(input,new Point(left[first_ind],right[first_ind]),new Point(left[second_ind],right[second_ind]),new Scalar(255,0,0,255),2,8,0);
            first_ind++;
            if(first_ind==size)
            {
                first_ind=0;
            }
            second_ind++;
            if(second_ind==size) {
                second_ind = 0;
            }
        }
    }

    public double get_mean(int [] left, int [] right, int index[])
    {
        int first_ind = index[0];
        int second_ind = index[0] + 1;
        int third_ind = index[0] + 2;


        int size = index[1];

        if(second_ind==size) {
            second_ind = 0;
            third_ind = 1 ;
        }

        if(third_ind==size) {
            third_ind = 0;
        }


        double sum_x_diff = 0;
        double sum_y_diff = 0;
        double sum_length = 0;

        for(int i=0;i<index[1]-1;i++)
        {
            double xdiff_1 = left[second_ind] - left[first_ind];
            double ydiff_1 = right[second_ind] - right[first_ind];


       //     double xdiff_2 = left[third_ind] - left[second_ind];
       //     double ydiff_2 = right[third_ind] - right[second_ind];




            double length1= hypot(xdiff_1,ydiff_1);
         //   double length2= hypot(xdiff_2,ydiff_2);




      //      xdiff_1 = xdiff_1 / length1;
      //      ydiff_1 = ydiff_1 / length1;

        //    xdiff_2 = xdiff_2 / length2;
        //    ydiff_2 = ydiff_2 / length2;

         //   xdiff_2 = xdiff_2 - xdiff_1;
         //   ydiff_2 = ydiff_2 - ydiff_1;

          //  double length_diff = hypot(xdiff_2,ydiff_2);;

            sum_length = sum_length+length1;


            first_ind++;
            if(first_ind==size)
            {
                first_ind=0;
            }
            second_ind++;
            if(second_ind==size) {
                second_ind = 0;
            }
          //  third_ind++;
          //  if(third_ind==size) {
         //       third_ind = 0;
         //   }
        }

        return sum_length/size;


    }

    public int get_car_number(Mat input)
    {
        MatOfRect carDetections = new MatOfRect();
        carDetector.detectMultiScale(input,carDetections);
        return carDetections.toArray().length;

    }

    public void PB_test_click(View view)
    {
        int input =3;
        int []a= new int[3];
        a[0]=4;
        testInt(a);
        TextView tv = (TextView) findViewById(R.id.TV_1);

        tv.setText(String.format("%d",a[0]));
    }

    public native void FindFeatures( long matAddrRgba, long matAddrRgba_prev);
    public native void SaveBoundles( long matAddrRgba, long matAddrRgba_prev, String path);
    public native void GetBiggestBoundle(long addrRgba, long addrRgba_prev,int[] nums);
    public native void TrackRect(long addrRgba, long addrRgba_prev,int[] nums);
    public native void FindInRect(long addrRgba, long addrRgba_prev,int[] nums);
    public native void MatchRect(long addrRgba, long addrRgba_prev,int[] nums, double[] values);
    public native void CreateMask(long addrRgba, long addrRgba_prev,long addrGr_output);
    public native void FindDenseArea(long addrMask,int[] left,int[] right,int[] index);
    public native void HasCar(long addrMask,int[] car_number);
    public native void GetClass(long addImg, String basic_path, int[] class_number);






    public native void testInt(int[] input);

    //
    public native void IncreaseContrast(long matAddrRgba);


    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public static native String stringFromJNI(String input,int[] nums);

    private void showToast(final String text) {
        final Activity activity = this;
        if (activity != null) {
            activity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(activity, text, Toast.LENGTH_SHORT).show();
                }
            });
        }
    }

    final Activity getActivity(){
        final Activity activity = this;
        return activity;
    }
}


//
//
//public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
//    int frame_count=0;
//    boolean first_frame = true;
//    Mat prev_gray;
//    Mat prev_rgba;
//    File basic_path = new File(Environment.getExternalStorageDirectory()+"/myImages/");
//
//
//    CameraBridgeViewBase.CvCameraViewFrame prev_frame;
//    private static final String TAG = "OCVSample::Activity";
//
//    private CameraBridgeViewBase mOpenCvCameraView;
//    private boolean              mIsJavaCamera = true;
//    private MenuItem mItemSwitchCamera = null;
//
//    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
//        @Override
//        public void onManagerConnected(int status) {
//            switch (status) {
//                case LoaderCallbackInterface.SUCCESS:
//                {
//                    Log.i(TAG, "OpenCV loaded successfully");
//                    mOpenCvCameraView.enableView();
//                } break;
//                default:
//                {
//                    super.onManagerConnected(status);
//                } break;
//            }
//        }
//    };
//
//    /** Called when the activity is first created. */
//    @Override
//    public void onCreate(Bundle savedInstanceState) {
//        Log.i(TAG, "called onCreate");
//        basic_path.mkdir();
//        super.onCreate(savedInstanceState);
//
//        // Load ndk built module, as specified
//        // in moduleName in build.gradle
//        System.loadLibrary("native");
//
//        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
//
//        setContentView(R.layout.activity_main);
//
//        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
//
//        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
//
//        mOpenCvCameraView.setCvCameraViewListener(this);
//    }
//
//    @Override
//    public void onPause()
//    {
//        super.onPause();
//        disableCamera();
//    }
//
//    @Override
//    public void onResume()
//    {
//        super.onResume();
//        if (!OpenCVLoader.initDebug()) {
//            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
//            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
//        } else {
//            Log.d(TAG, "OpenCV library found inside package. Using it!");
//            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//        }
//    }
//
//    public void onDestroy() {
//        super.onDestroy();
//        disableCamera();
//    }
//
//    public void disableCamera() {
//        if (mOpenCvCameraView != null)
//            mOpenCvCameraView.disableView();
//    }
//
//    public void onCameraViewStarted(int width, int height) {
//    }
//
//    public void onCameraViewStopped() {
//    }
//
//    Mat mRgba;
//    Mat mGray;
//
//    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//        Mat temp = performFindFeatures(inputFrame);
//
//
//       // File file = new File(getActivity().getExternalFilesDir(null),Integer.toString(frame_count++)+".png");
//
//
//        String filename = basic_path.toString()+"/"+Integer.toString(frame_count++)+".png";
//        boolean result = Imgcodecs.imwrite(filename,temp);
//        if(result)
//            showToast(filename);
//        else
//            showToast("error");
//
//        //showToast("Saved: " + file);
//         return temp;
//        //return performIncreaseContrast(inputFrame);
//    }
//
//    protected Mat performFindFeatures(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//        if(first_frame)
//        {
//            first_frame = false;
//            prev_gray = Mat.zeros(inputFrame.gray().size(),inputFrame.gray().type());
//            prev_rgba = Mat.zeros(inputFrame.rgba().size(),inputFrame.rgba().type());
//            inputFrame.gray().copyTo(prev_gray);
//            inputFrame.rgba().copyTo(prev_rgba);
//        }
//        mRgba = inputFrame.rgba();
//        mGray = inputFrame.gray();
//
//        Mat temp_gray = Mat.zeros(inputFrame.gray().size(),inputFrame.gray().type());
//        Mat temp_rgba = Mat.zeros(inputFrame.rgba().size(),inputFrame.rgba().type());
//        inputFrame.gray().copyTo(temp_gray);
//        inputFrame.rgba().copyTo(temp_rgba);
//
//        FindFeatures(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr(), prev_gray.getNativeObjAddr(), prev_rgba.getNativeObjAddr());
//
//        temp_gray.copyTo(prev_gray);
//        temp_rgba.copyTo(prev_rgba);
//
//        temp_gray.release();
//        temp_rgba.release();
//        //prev_gray = inputFrame.gray();
//        //prev_rgba = inputFrame.rgba();
//        return mRgba;
//    }
//
//    protected Mat performIncreaseContrast(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//        mRgba = inputFrame.rgba();
//        IncreaseContrast(mRgba.getNativeObjAddr());
//        return mRgba;
//    }
//
//    public native void FindFeatures(long matAddrGr, long matAddrRgba, long matAddrGr_prev, long matAddrRgba_prev);
//
//    public native void IncreaseContrast(long matAddrRgba);
//
//    private void showToast(final String text) {
//        final Activity activity = this;
//        if (activity != null) {
//            activity.runOnUiThread(new Runnable() {
//                @Override
//                public void run() {
//                    Toast.makeText(activity, text, Toast.LENGTH_SHORT).show();
//                }
//            });
//        }
//    }
//    final Activity getActivity(){
//        final Activity activity = this;
//        return activity;
//    }
//}
