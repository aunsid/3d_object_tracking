
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::Mat duplicate;
    cv::resize(topviewImg, duplicate, cv::Size(), 0.5, 0.5);
    cv::namedWindow(windowName, 1);
    // cv::imshow(windowName, topviewImg);
    cv::imshow(windowName, duplicate);
    // cv::resizeWindow(windowName,640,480);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev, 
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> matches;
    // iterate over matches
    for (const auto &match :kptMatches)
    {
        if(boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            matches.push_back(match);
        }
    }
    

    // eliminate outliers
    double mean_distance = 0;
    if (matches.size() ==0)
    {
        return;
    }
    else
    {
        for(const auto & match: matches)
        {
            mean_distance = mean_distance + match.distance;
        }
        mean_distance = mean_distance / matches.size();
        double threshold = mean_distance * 0.75;

        for (const auto &match: matches)
        {
            if(match.distance < threshold)
            {
                boundingBox.kptMatches.push_back(match);
            }
        }

    }

    
    
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches,
                      double frameRate,
                      double &TTC,
                      cv::Mat *visImg)
{   
    std::vector<double> distRatios;

    for (auto it1 = kptMatches.begin(); it1!= kptMatches.end()-1; it1++)
    {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin()+1; it2!=kptMatches.end(); it2++)
        {   
            double minDist =100;
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);


            if(distPrev> std::numeric_limits<double>::epsilon()&& distCurr>=minDist)
            {
                double distRatio = distCurr/distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    if(distRatios.size() == 0)
    {
        TTC =NAN;
        return;
    }

    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0)/distRatios.size();
    double dT = 1/ frameRate;
    TTC = -dT /(1-meanDistRatio);

    std::cout<<"Camera TTC  "<< TTC<<std::endl;
}

double getMedian (std::vector<double> points)
{   
    size_t size= points.size();

    if (size == 0)
    {
        return 0;
    }
    else
    {
        std::sort(points.begin(), points.end());
        if(size%2 ==0)
            return (points[size/2 -1]+ points[size/2])/2;

        else 
            return (points[size/2]);
    }
    
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
     

    std::vector<double> prevPointX;
    std::vector<double> currPointX;

    double prevMedX = 0.0;
    double currMedX = 0.0;

    // get the median of prev lidar points
    for (auto point:lidarPointsPrev)
    {
        prevPointX.push_back(point.x);
    }

    prevMedX = getMedian(prevPointX);
    
    // get the median of curr lidar points

    for (auto point:lidarPointsCurr)
    {
        currPointX.push_back(point.x);
    }

    currMedX = getMedian(currPointX);


    double dt = 1/ frameRate;

    TTC = currMedX * dt/ (prevMedX - currMedX);

    std::cout<<"Lidar TTC  "<< TTC<<std::endl;
}




void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
   


    int p = prevFrame.boundingBoxes.size();
    int c = currFrame.boundingBoxes.size();

    int pts [p][c] = {};
    for(auto it = matches.begin(); it != matches.end(); it++)
    {   
        // get points in the matches
        cv::KeyPoint prevPoint = prevFrame.keypoints[it->queryIdx];
        cv::KeyPoint currPoint = currFrame.keypoints[it->trainIdx];

        // record the box id where the keypoints are located
        int pID = -1;
        int cID = -1;
        for( auto pbox: prevFrame.boundingBoxes)
        {
            if(pbox.roi.contains(prevPoint.pt))
                pID = pbox.boxID;

        }
        for(auto cbox: currFrame.boundingBoxes)
        {
            if(cbox.roi.contains(currPoint.pt))
                cID = cbox.boxID;
                   

        }

        if(pID !=-1 && cID!=-1)
        {
            pts[pID][cID] +=1;
        }
        
    }

    for (int i= 0; i< p; i++)
    {
        int max_count =0;
        int id_max =0;
        for (int j=0; j<c; j++)
        {
            if(pts[i][j]> max_count)
            {
                max_count = pts[i][j];
                id_max = j;
            }
        }
        bbBestMatches[i] = id_max;

    }
    
}
