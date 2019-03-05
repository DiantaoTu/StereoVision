#include <opencv2/opencv.hpp>
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <stdio.h>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

Mat SgbmTest(Mat left_ud,Mat right_ud);             //使用SGBM算法进行立体重构
void On_mouse(int event,int x,int y,int flag, void* data);    //鼠标单机时的操作

int main(int argc,char** argv){
    VideoCapture capture(0);                        //有时候是0有时候是1，奇了怪了
    Mat frame;                                      //相机获取的一帧
    Mat left,right;                                 //左右图像
    Mat left_ud,right_ud;                           //经过畸变矫正的左右图像
    Mat camera_left = Mat::eye(3,3,CV_64F);         //左侧相机内参数矩阵
    Mat camera_right = Mat::eye(3,3,CV_64F);        //右侧相机内参数矩阵
    Mat dist_coeffs_left = Mat::eye(5,1,CV_64F);    //左侧相机畸变参数向量
    Mat dist_coeffs_right = Mat::eye(5,1,CV_64F);   //右侧相机畸变参数向量
    Mat R = Mat::eye(3,1,CV_64F);
    Mat t = Mat::eye(3,1,CV_64F);
    Mat map1,map2,map3,map4,R1,R2,P1,P2,Q;
    Rect rio1,rio2;
    Size img_size;

    camera_left.at<double>(0, 0) = 406.86448; 
    camera_left.at<double>(0, 2) = 149.08732; 
    camera_left.at<double>(1, 1) = 407.91008; 
    camera_left.at<double>(1, 2) = 118.67917;

    camera_right.at<double>(0, 0) = 404.99717; 
    camera_right.at<double>(0, 2) = 152.64756; 
    camera_right.at<double>(1, 1) = 407.34931; 
    camera_right.at<double>(1, 2) = 121.40527;

    dist_coeffs_left.at<double>(0, 0) = -0.41906;
	dist_coeffs_left.at<double>(1, 0) = -0.05569;
	dist_coeffs_left.at<double>(2, 0) = -0.00108;
	dist_coeffs_left.at<double>(3, 0) = 0.00425;
    dist_coeffs_left.at<double>(4, 0) = 0;

    dist_coeffs_right.at<double>(0, 0) = -0.50873;
	dist_coeffs_right.at<double>(1, 0) = 0.51490;
	dist_coeffs_right.at<double>(2, 0) = 0.00193;
	dist_coeffs_right.at<double>(3, 0) = -0.01215;
    dist_coeffs_right.at<double>(4, 0) = 0;

    R.at<double>(0, 0) = 0.01800;
	R.at<double>(1, 0) = -0.01484;
	R.at<double>(2, 0) = 0.00276;

	t.at<double>(0, 0) = -172.89628;
	t.at<double>(1, 0) = 1.79758;
    t.at<double>(2, 0) = 4.28381;




/*
    camera_left.at<double>(0, 0) = 408.74870;
	camera_left.at<double>(0, 2) = 157.13605;
	camera_left.at<double>(1, 1) = 408.90549;
	camera_left.at<double>(1, 2) = 119.50782;

	dist_coeffs_left.at<double>(0, 0) = -0.36491;
	dist_coeffs_left.at<double>(1, 0) = 0.00182;
	dist_coeffs_left.at<double>(2, 0) = 0.00331;
	dist_coeffs_left.at<double>(3, 0) = -0.00080;
	dist_coeffs_left.at<double>(4, 0) = 0;

	camera_right.at<double>(0, 0) = 400.19771;
	camera_right.at<double>(0, 2) = 147.69264;
	camera_right.at<double>(1, 1) = 400.46567;
	camera_right.at<double>(1, 2) = 101.55715;

	dist_coeffs_right.at<double>(0, 0) = -0.36531;
	dist_coeffs_right.at<double>(1, 0) = 0.08399;
	dist_coeffs_right.at<double>(2, 0) = 0.00362;
	dist_coeffs_right.at<double>(3, 0) = 0.00456;
	dist_coeffs_right.at<double>(4, 0) = 0;

	R.at<double>(0, 0) = -0.04235;
	R.at<double>(1, 0) = 0.00127;
	R.at<double>(2, 0) = -0.00054;

	t.at<double>(0, 0) = -167.03475;
	t.at<double>(1, 0) = -0.32197;
	t.at<double>(2, 0) = -3.69357;
*/
    //处理第一帧画面，获取一些后面所需的基本参数
    capture>>frame;
    int col = frame.cols;
    int row = frame.rows;
     
    left = frame(Rect(0, 0, col/2, row));       //分割为左右画面
    right = frame(Rect(col/2, 0, col/2, row));  
    img_size = left.size();
    
    Mat R_R;
    Rodrigues(R,R_R);   //使用罗德里格斯公式把R变为旋转矩阵			
    //输入双目的内参等，输出R1 R2 P1 P2 Q roi1 roi2 用于矫正畸变和计算深度	
	stereoRectify(camera_left, dist_coeffs_left, camera_right, dist_coeffs_right, img_size, R_R, t, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &rio1, &rio2);

	initUndistortRectifyMap(camera_left, dist_coeffs_left, R1, P1, img_size, CV_16SC2, map1, map2);	

    initUndistortRectifyMap(camera_right, dist_coeffs_right, R2, P2, img_size, CV_16SC2, map3, map4); 

    chrono::steady_clock::time_point start,end;
    chrono::duration<double> time_used;
    Mat disp, xyz, depth;           //disp是视差图，xyz是视差图在相机坐标系下的坐标，depth是视差图每一点的深度
    vector<Mat> xyz_split;          //xyz经过通道分离后存储在这里
    while(true){
        start = chrono::steady_clock::now();

        capture>>frame;
        left = frame(Rect(0, 0, col/2, row));       //分割为左右画面
        right = frame(Rect(col/2, 0, col/2, row));

        remap(left,left_ud,map1,map2,INTER_LINEAR);
        remap(right,right_ud,map3,map4,INTER_LINEAR);

        imshow("left_ud",left_ud);
        imshow("right_ud",right_ud);

        cvtColor(left_ud,left_ud,CV_BGR2GRAY);      //转换成灰度图
        cvtColor(right_ud,right_ud,CV_BGR2GRAY);

        disp = SgbmTest(left_ud,right_ud);     //使用SGBM方法进行深度分析,视差矩阵作为返回值

        reprojectImageTo3D(disp,xyz,Q);         //生成视差图在像极坐标系下的xyz坐标
        xyz *= 16;                  //disp 是CV_16S类型的，出于精度需要扩大了16倍，结果要乘以16获得mm级精度
        xyz.convertTo(xyz,CV_64FC3);
        // split(xyz,xyz_split);
        // depth = xyz_split[2];
        
        setMouseCallback("left_ud",On_mouse,&xyz);
            

        end = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(end-start);
        //cout<<"每一帧用时："<<time_used.count()<<"秒"<<endl;
        //使用waitkey时每一帧耗时0.015秒左右  不使用的时候耗时0.03秒左右  为啥
        waitKey(time_used.count()*1000);   
         
    }
}

//详细参数设置见https://blog.csdn.net/cxgincsu/article/details/74451940

Mat SgbmTest(Mat left_ud,Mat right_ud){ 
    int SADWindowSize =7;//必须是奇数
    Ptr<StereoSGBM> sgbm = StereoSGBM::create();
    sgbm->setBlockSize(SADWindowSize);
    sgbm->setP1(8 *1*SADWindowSize*SADWindowSize);
	sgbm->setP2(32 *1*SADWindowSize*SADWindowSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(128);//128//num_disp good
	sgbm->setUniquenessRatio(5);//good
	sgbm->setSpeckleWindowSize(100);//good
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setPreFilterCap(64);// good
    sgbm->setMode(StereoSGBM::MODE_HH);//good
    Mat disp,disp_8;
    sgbm->compute(left_ud,right_ud,disp);

    disp = disp.colRange(128,disp.cols);        //转换后左侧有一部分没有用，删去
    disp.convertTo(disp_8,CV_8U,255/(128*16.));

    imshow("disp",disp_8);
    return disp;

}

void On_mouse(int event, int x, int y, int flage, void* data){
    Mat xyz;
    xyz = *(Mat*)data;
    
    static vector<double> d;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:{
            cout<<"像素坐标 x "<<x<<" y "<<y<<endl;
            if(x<128){
                cout<<"没有对应的世界坐标"<<endl;
                break;
            }
            cout<<" x "<<xyz.at<Vec3d>(x-128,y)[0];
            cout<<" y "<<xyz.at<Vec3d>(x-128,y)[1];
            cout<<" z "<<xyz.at<Vec3d>(x-128,y)[2]<<endl;
            // d.push_back(depth.at<double>(p));
            // double sum = std::accumulate(d.begin(),d.end(),0.0);
            // double avg = sum/d.size();
            // double accum  = 0.0;
	        // std::for_each (std::begin(d), std::end(d), [&](const double tmp) {
		    //     accum  += (tmp-avg)*(tmp-avg);
	        // });
            // double stdev = sqrt(accum/(d.size()-1)); //方差
            // cout<<"平均值: "<<avg<<endl;
            // cout<<"方差:   "<<stdev<<endl;
            break;
        }
            
    
        default:
            break;
    }

}
