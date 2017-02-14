package org.honorato.opencvsample;

import android.app.Activity;
import android.content.res.AssetManager;
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

import com.sh1r0.caffe_android_lib.CaffeMobile;



import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import static java.lang.Math.hypot;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC1;


public class MainActivity extends AppCompatActivity {

    File basic_path = new File(Environment.getExternalStorageDirectory()+"/myImages");
    String dataset_name = "HKUST1";

    private static final int REQUEST_IMAGE_CAPTURE = 100;
    private static final int REQUEST_IMAGE_SELECT = 200;
    public static final int MEDIA_TYPE_IMAGE = 1;
    private static String[] IMAGENET_CLASSES;
    private CaffeMobile caffeMobile;

    static {
        System.loadLibrary("caffe");
        System.loadLibrary("caffe_jni");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        System.loadLibrary("native");

        //basic_path.mkdir();
        caffeMobile = new CaffeMobile();
        caffeMobile.setNumThreads(4);
        String model_path = basic_path + "/model";
        File test= new File(model_path+"/test2");
        test.mkdir();
        //caffeMobile.loadModel(model_path+"/bvlc_reference_caffenet/deploy.prototxt",
        //        model_path+"/bvlc_reference_caffenet/1.caffemodel");

        caffeMobile.loadModel(model_path+"/new_model/deploy.prototxt",
                model_path+"/new_model/1.caffemodel");

        float[] meanValues = {104, 117, 123};
        caffeMobile.setMean(meanValues);

        AssetManager am = this.getAssets();
        try {
            InputStream is = am.open("synset_words.txt");
            Scanner sc = new Scanner(is);
            List<String> lines = new ArrayList<String>();
            while (sc.hasNextLine()) {
                final String temp = sc.nextLine();
                lines.add(temp.substring(temp.indexOf(" ") + 1));
            }
            IMAGENET_CLASSES = lines.toArray(new String[0]);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    public void PB_caffetest_click(View view) {
        String img_path = basic_path+"/danial.jpg";
        TextView tv = (TextView) findViewById(R.id.TV_1);
        int[] ret = caffeMobile.predictImage(img_path);

        tv.setText(String.format("return:%d ",ret[0])+IMAGENET_CLASSES[ret[0]]);
        return;
    }

    public void PB_transform_click(View view)
    {
        showToast("start");
        TextView tv = (TextView) findViewById(R.id.TV_1);
        EditText ET_start = (EditText) findViewById(R.id.ET_start);
        EditText ET_end = (EditText) findViewById(R.id.ET_end);
        String images_folder = basic_path + "/HKUST1";
        String out_folder = basic_path + "/HKUST1/HKUST1_result";
        File out_file = new File(out_folder);
        out_file.mkdir();
        Mat input,prev,output,normal;
        prev = Mat.eye(2, 2, CV_8U);
        int start_frame = (int) Long.parseLong(ET_start.getText().toString());
        int end_frame = (int) Long.parseLong(ET_end.getText().toString());
        boolean tracking = false;
        Rect tracking_bound = new Rect(0, 0, 3, 3);
        String min_values = "";


        int[] rect_data = new int[5];
        int[] left_point = new int[10];
        int[] right_point = new int[10];
        int[] index = new int[2];

        int[] class_number = new int[1];

        index[1] = 10; //size of circular buffer


        int resize_cols, resize_rows;

        input = Imgcodecs.imread(images_folder + "/images/" + String.format("%010d", start_frame) + ".png");
        resize_cols = input.cols() / 4;
        resize_rows = input.rows() / 4;
        output = Mat.zeros(resize_rows, resize_cols, CV_8UC1);
        normal = Mat.zeros(resize_rows, resize_cols, CV_8UC1);

        int counter = 0;

        long sum_mask = 0, sum_svm_class = 0, sum_cnn_class = 0, sum_all = 0;
        long mask_counter = 0, svm_class_counter = 0, cnn_class_counter = 0, all_counter = 0;


        for (int i = start_frame + 1; i < end_frame; i++)
        {
            Mat input_resized;
            input = Imgcodecs.imread(images_folder + "/images/" + String.format("%010d", i) + ".png");
            resize_cols = input.cols() / 4;
            resize_rows = input.rows() / 4;
            input_resized = Mat.zeros(resize_rows, resize_cols, input.type());

            Size size2 = new Size(resize_cols, resize_rows);
            Imgproc.resize(input, input_resized, size2);


            if (i == start_frame + 1)
            {
                prev = Mat.zeros(input_resized.size(), input_resized.type());
                input_resized.copyTo(prev);
            }
            else
            {
                counter++;
                Mat temp, tosave;
                long MaskStartTime = System.nanoTime();
                long AllStartTime = System.nanoTime();
                temp = Mat.zeros(input_resized.size(), input_resized.type());
                input_resized.copyTo(temp);
                //output = Mat.zeros(resize_rows,resize_cols,CV_32FC1);
                CreateMask2(input_resized.getNativeObjAddr(), prev.getNativeObjAddr(), output.getNativeObjAddr());
                Imgcodecs.imwrite(out_folder + "/image_sub_" + String.format("%08d", i) + "_cnv.png", output);
                Core.normalize(output, output, 0, 127, Core.NORM_MINMAX);

                //output.convertTo(output,CV_32FC1);
                Core.add(output, normal, normal);
                Imgproc.dilate(normal, normal, Mat.ones(2, 8, CV_8UC1));
                Core.normalize(normal, normal, 0, 255, Core.NORM_MINMAX);
                tosave = Mat.zeros(normal.size(), CV_8UC1);
                Core.normalize(normal, normal, 0, 127, Core.NORM_MINMAX);
                normal.copyTo(tosave);
                Imgcodecs.imwrite(out_folder + "/image_norm_" + String.format("%08d", i) + "_cnv.png", tosave);
                FindDenseArea(normal.getNativeObjAddr(), left_point, right_point, index);
                if (counter > index[1])//buffer has been filled
                {
                    Core.normalize(tosave, tosave, 0, 200, Core.NORM_MINMAX);
                    draw_points(tosave, left_point, right_point, index);
                    Imgcodecs.imwrite(out_folder + "/image_points_" + String.format("%08d", i) + "_cnv.png", tosave);
                    double mean = get_mean(left_point, right_point, index);
                    long MaskEndTime = System.nanoTime();
                    long MaskTime = MaskEndTime - MaskStartTime;
                    sum_mask = sum_mask + MaskTime / 1000000;
                    mask_counter = mask_counter + 1;

                    //Double.toString(mean)
                    //  Imgproc.putText(tosave,Double.toString(mean),new Point(50,50),3,(double) 2,new Scalar(255,0,0,255),3);
                 //  for(int mean_th=5;mean_th<10;mean_th++)
                    int mean_th=7;
                   {
                       if (mean < mean_th) {

                           Rect bound = get_proposal_rect(input_resized, left_point, right_point, index);
                           Mat mat_bound = new Mat(input, new Rect(bound.x * 4, bound.y * 4, bound.width * 4, bound.height * 4));

                           long svmStartTime = System.nanoTime();
                         //  GetClass(mat_bound.getNativeObjAddr(), basic_path.toString(), class_number);
                           long svmEndTime = System.nanoTime();
                           long svmTime = svmEndTime - svmStartTime;

                           long cnnStartTime = System.nanoTime();
                           int caffe_class = get_caffe_class(mat_bound);
                           //Imgcodecs.imwrite(out_folder + "/cropped_image_" + String.format("%08d", i) + "_cnv.png", mat_bound);
                           // int [] ret = caffeMobile.predictImage(out_folder + "/cropped_image_" + String.format("%08d", i) + "_cnv.png");
                           //int caffe_class = ret[0];
                           long cnnEndTime = System.nanoTime();
                           long cnnTime = cnnEndTime - cnnStartTime;

                           Imgproc.rectangle(input_resized, bound.tl(), bound.br(), new Scalar(255, 0, 0, 255), 2, 8, 0);
                           //Imgproc.putText(input_resized, String.format("m:%d c:%d cf:%d", (int) mean, class_number[0],caffe_class), new Point(5,20), 3, (double) 1, new Scalar(255, 0, 0, 255), 3);
                           if(caffe_class==2)
                               Imgproc.putText(input_resized,"class: car", new Point(5,20), 3, (double) 1, new Scalar(255, 0, 0, 255), 3);
                           else
                               Imgproc.putText(input_resized,"class: not car", new Point(5,20), 3, (double) 1, new Scalar(255, 0, 0, 255), 3);

                           sum_svm_class = sum_svm_class + svmTime / 1000000;
                           sum_cnn_class = sum_cnn_class + cnnTime / 1000000;

                           svm_class_counter = svm_class_counter + 1;
                           cnn_class_counter = cnn_class_counter + 1;

                           mat_bound.release();


                       }


                   }
                    boolean result = Imgcodecs.imwrite(out_folder + "/image_" + String.format("th_%d_%08d",mean_th, i) + ".png", input_resized);


                }

                long AllEndTime = System.nanoTime();
                long AllTime = AllEndTime - AllStartTime;
                sum_all = sum_all + AllTime / 1000000;
                all_counter = all_counter + 1;



                temp.copyTo(prev);
                temp.release();
                tosave.release();
            }
            input_resized.release();
        }
        input.release();
        prev.release();
        output.release();
        normal.release();
        tv.setText(String.format("mask:%f svm:%f cnn:%f all:%f", (float) sum_mask / mask_counter, (float) sum_svm_class / svm_class_counter,(float) sum_cnn_class / cnn_class_counter, (float) sum_all / all_counter));

        showToast("end");
    }

    public void PB_start_algorithm_click(View view)
    {
        showToast("start");
        TextView tv = (TextView) findViewById(R.id.TV_1);
        EditText ET_start = (EditText) findViewById(R.id.ET_start);
        EditText ET_end = (EditText) findViewById(R.id.ET_end);
        String images_folder = basic_path + "/" + dataset_name;
        String out_folder = basic_path + "/" + dataset_name + "/results";
        File out_file = new File(out_folder);
        out_file.mkdir();
        Mat input,prev,output,normal;
        prev = Mat.eye(2, 2, CV_8U);
        int start_frame = (int) Long.parseLong(ET_start.getText().toString());
        int end_frame = (int) Long.parseLong(ET_end.getText().toString());
        boolean tracking = false;
        Rect tracking_bound = new Rect(0, 0, 3, 3);
        String min_values = "";


        int[] rect_data = new int[5];
        int[] left_point = new int[10];
        int[] right_point = new int[10];
        int[] index = new int[2];

        int[] class_number = new int[1];

        index[1] = 10; //size of circular buffer


        int resize_cols, resize_rows;

        input = Imgcodecs.imread(images_folder + "/images/" + String.format("%010d", start_frame) + ".png");
        resize_cols = input.cols() / 4;
        resize_rows = input.rows() / 4;
        output = Mat.zeros(resize_rows, resize_cols, CV_8UC1);
        normal = Mat.zeros(resize_rows, resize_cols, CV_8UC1);

        int counter = 0;

        long sum_mask = 0, sum_svm_class = 0, sum_cnn_class = 0, sum_all = 0;
        long mask_counter = 0, svm_class_counter = 0, cnn_class_counter = 0, all_counter = 0;


        for (int i = start_frame + 1; i < end_frame; i++)
        {
            Mat input_resized;
            input = Imgcodecs.imread(images_folder + "/images/" + String.format("%010d", i) + ".png");
            resize_cols = input.cols() / 4;
            resize_rows = input.rows() / 4;
            input_resized = Mat.zeros(resize_rows, resize_cols, input.type());

            Size size2 = new Size(resize_cols, resize_rows);
            Imgproc.resize(input, input_resized, size2);


            if (i == start_frame + 1)
            {
                prev = Mat.zeros(input_resized.size(), input_resized.type());
                input_resized.copyTo(prev);
            }
            else
            {
                counter++;
                Mat temp;
                long MaskStartTime = System.nanoTime();
                long AllStartTime = System.nanoTime();
                temp = Mat.zeros(input_resized.size(), input_resized.type());
                input_resized.copyTo(temp);
                //output = Mat.zeros(resize_rows,resize_cols,CV_32FC1);
                CreateMask2(input_resized.getNativeObjAddr(), prev.getNativeObjAddr(), output.getNativeObjAddr());
                Core.normalize(output, output, 0, 127, Core.NORM_MINMAX);

                //output.convertTo(output,CV_32FC1);
                Core.add(output, normal, normal);
                Imgproc.dilate(normal, normal, Mat.ones(2, 8, CV_8UC1));
                Core.normalize(normal, normal, 0, 127, Core.NORM_MINMAX);
                FindDenseArea(normal.getNativeObjAddr(), left_point, right_point, index);
                if (counter > index[1])//buffer has been filled
                {
                    double mean = get_mean(left_point, right_point, index);
                    long MaskEndTime = System.nanoTime();
                    long MaskTime = MaskEndTime - MaskStartTime;
                    sum_mask = sum_mask + MaskTime / 1000000;
                    mask_counter = mask_counter + 1;

                    //  Double.toString(mean)
                    //  Imgproc.putText(tosave,Double.toString(mean),new Point(50,50),3,(double) 2,new Scalar(255,0,0,255),3);
                    //  for(int mean_th=5;mean_th<10;mean_th++)
                    int mean_th=7;
                    {
                        if (mean < mean_th) {

                            Rect bound = get_proposal_rect(input_resized, left_point, right_point, index);
                            Mat mat_bound = new Mat(input, new Rect(bound.x * 4, bound.y * 4, bound.width * 4, bound.height * 4));

                            long svmStartTime = System.nanoTime();
                            //  GetClass(mat_bound.getNativeObjAddr(), basic_path.toString(), class_number);
                            long svmEndTime = System.nanoTime();
                            long svmTime = svmEndTime - svmStartTime;

                            long cnnStartTime = System.nanoTime();
                            int caffe_class = get_caffe_class(mat_bound);
                            //Imgcodecs.imwrite(out_folder + "/cropped_image_" + String.format("%08d", i) + "_cnv.png", mat_bound);
                            // int [] ret = caffeMobile.predictImage(out_folder + "/cropped_image_" + String.format("%08d", i) + "_cnv.png");
                            //int caffe_class = ret[0];
                            long cnnEndTime = System.nanoTime();
                            long cnnTime = cnnEndTime - cnnStartTime;

                            Imgproc.rectangle(input_resized, bound.tl(), bound.br(), new Scalar(255, 0, 0, 255), 2, 8, 0);
                            //Imgproc.putText(input_resized, String.format("m:%d c:%d cf:%d", (int) mean, class_number[0],caffe_class), new Point(5,20), 3, (double) 1, new Scalar(255, 0, 0, 255), 3);
                            if(caffe_class==2)
                                Imgproc.putText(input_resized,"class: car", new Point(5,20), 3, (double) 1, new Scalar(255, 0, 0, 255), 3);
                            else
                                Imgproc.putText(input_resized,"class: not car", new Point(5,20), 3, (double) 1, new Scalar(255, 0, 0, 255), 3);

                            sum_svm_class = sum_svm_class + svmTime / 1000000;
                            sum_cnn_class = sum_cnn_class + cnnTime / 1000000;

                            svm_class_counter = svm_class_counter + 1;
                            cnn_class_counter = cnn_class_counter + 1;

                            mat_bound.release();


                        }


                    }
                    boolean result = Imgcodecs.imwrite(out_folder + "/image_" + String.format("th_%d_%08d",mean_th, i) + ".png", input_resized);


                }

                long AllEndTime = System.nanoTime();
                long AllTime = AllEndTime - AllStartTime;
                sum_all = sum_all + AllTime / 1000000;
                all_counter = all_counter + 1;



                temp.copyTo(prev);
                temp.release();
              //  tosave.release();
            }
            input_resized.release();
        }
        input.release();
        prev.release();
        output.release();
        normal.release();
        tv.setText(String.format("mask:%f cnn:%f all:%f", (float) sum_mask / mask_counter, (float) sum_cnn_class / cnn_class_counter, (float) sum_all / all_counter));

        showToast("end");
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

    int get_caffe_class(Mat img)
    {
        String temp_path = basic_path+"/temp.jpg";
        Imgcodecs.imwrite(temp_path,img);
        int [] ret = caffeMobile.predictImage(temp_path);
        return ret[0];

    }

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



    public native void FindFeatures(long matAddrGr, long matAddrRgba);

    public native void IncreaseContrast(long matAddrRgba);
    public static native String stringFromJNI(String input,int[] nums);
    public native void CreateMask(long addrRgba, long addrRgba_prev,long addrGr_output);
    public native void CreateMask2(long addrRgba, long addrRgba_prev,long addrGr_output);

    public native void FindDenseArea(long addrMask,int[] left,int[] right,int[] index);
    public native void GetClass(long addImg, String basic_path, int[] class_number);
}
