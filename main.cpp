#include <string>

#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/conversions.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/pcl_visualizer.h>

using PointType = pcl::PointXYZ;
using PointCloudXYZ = pcl::PointCloud<PointType>;
using NormalCloud = pcl::PointCloud<pcl::Normal>;
PointCloudXYZ::Ptr src, tgt;

void drawPointCloud(PointCloudXYZ::Ptr pc1, PointCloudXYZ::Ptr pc2, std::string stageName)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Stage: " + stageName));
    viewer->setBackgroundColor(0, 0, 0);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc1_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pc1_rgb->points.resize(pc1->size());
    for (size_t i = 0; i < pc1->points.size(); i++) {
	pc1_rgb->points[i].x = pc1->points[i].x;
	pc1_rgb->points[i].y = pc1->points[i].y;
	pc1_rgb->points[i].z = pc1->points[i].z;
	pc1_rgb->points[i].r = 255;
	pc1_rgb->points[i].g = 0;
	pc1_rgb->points[i].b = 0;
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(pc1_rgb);
    viewer->addPointCloud<pcl::PointXYZRGB>(pc1_rgb, rgb1, "First cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "First cloud");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc2_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pc2_rgb->points.resize(pc2->size());
    for (size_t i = 0; i < pc2->points.size(); i++) {
	pc2_rgb->points[i].x = pc2->points[i].x;
	pc2_rgb->points[i].y = pc2->points[i].y;
	pc2_rgb->points[i].z = pc2->points[i].z;
	pc2_rgb->points[i].r = 0;
	pc2_rgb->points[i].g = 255;
	pc2_rgb->points[i].b = 0;
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(pc2_rgb);
    viewer->addPointCloud<pcl::PointXYZRGB>(pc2_rgb, rgb2, "Second cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Second cloud");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped()) {
	viewer->spinOnce(360);
	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

template <typename FeatureType>
void findCorrespondences(const typename pcl::PointCloud<FeatureType>::Ptr& fpfhs_src,
    const typename pcl::PointCloud<FeatureType>::Ptr& fpfhs_tgt,
    pcl::Correspondences& all_correspondences)
{
    pcl::registration::CorrespondenceEstimation<FeatureType, FeatureType> est;
    est.setInputSource(fpfhs_src);
    est.setInputTarget(fpfhs_tgt);
    est.determineReciprocalCorrespondences(all_correspondences);
}

////////////////////////////////////////////////////////////////////////////////
void rejectBadCorrespondences(const pcl::CorrespondencesPtr& all_correspondences,
    const PointCloudXYZ::Ptr& keypoints_src,
    const PointCloudXYZ::Ptr& keypoints_tgt,
    pcl::Correspondences& remaining_correspondences)
{
    pcl::registration::CorrespondenceRejectorDistance rej;
    rej.setInputSource<PointType>(keypoints_src);
    rej.setInputTarget<PointType>(keypoints_tgt);
    rej.setInputCorrespondences(all_correspondences);
    rej.getCorrespondences(remaining_correspondences);
}

#include <pcl/registration/icp.h>
Eigen::Matrix4f getTransform_icp(PointCloudXYZ::Ptr source_cloud, PointCloudXYZ::Ptr target_cloud)
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    //icp.setMaxCorrespondenceDistance(0.1);

    PointCloudXYZ Final;
    icp.align(Final);
    std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    std::cout << "ICP transformation:" << icp.getFinalTransformation() << std::endl;
    return icp.getFinalTransformation();
}

boost::shared_ptr<PointCloudXYZ> estimateKeypoints(PointCloudXYZ::ConstPtr cloud)
{

    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud(cloud);
    //TODO automatically adjust the search radius. This could be done by binary searching for the radius until the two point clouds have (close to) equal sizes
    uniform_sampling.setRadiusSearch(1);
    boost::shared_ptr<PointCloudXYZ> keypoints_cloud(new PointCloudXYZ);
    uniform_sampling.filter(*keypoints_cloud);
    std::cout << "Found " << keypoints_cloud->size() << " keypoints" << std::endl;

    return keypoints_cloud;
}

NormalCloud::Ptr estimateNormals(PointCloudXYZ::ConstPtr cloud)
{
    pcl::NormalEstimation<PointType, pcl::Normal> normal_est;
    normal_est.setInputCloud(cloud);
    normal_est.setSearchMethod(pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>));
    normal_est.setKSearch(20);
    NormalCloud::Ptr normals(new NormalCloud);
    normal_est.compute(*normals);

    //check if the normals are finite
    for (int i = 0; i < normals->points.size(); i++) {
	if (!pcl::isFinite<pcl::Normal>(normals->points[i])) {
	    PCL_WARN("normals[%d] is not finite\n", i);
	}
    }
    return normals;
}

//using FeatureType = pcl::SHOT352;
using FeatureType = pcl::FPFHSignature33;
using FeaturePointCloud = pcl::PointCloud<FeatureType>;
FeaturePointCloud::Ptr estimateFPFH(PointCloudXYZ::ConstPtr cloud, NormalCloud::ConstPtr normals)
{
    pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> est;
    std::cout << "Entered" << std::endl;
    //pcl::SHOTEstimation<PointType, pcl::Normal, pcl::SHOT352> est;
    est.setInputCloud(cloud);
    est.setInputNormals(normals);
    est.setSearchMethod(pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>));
    est.setKSearch(40);
    //est.setRadiusSearch(10);
    //est.setSearchSurface(cloud);
    //fpfh_est.setRadiusSearch(0.1);

    FeaturePointCloud::Ptr features(new FeaturePointCloud);
    est.compute(*features);

    std::cout << "features:" << features->size() << std::endl;

    return features;
}

Eigen::Matrix4f getTransform(PointCloudXYZ::Ptr source, PointCloudXYZ::Ptr target)
{
    PointCloudXYZ::Ptr keypoints_source = estimateKeypoints(source);
    PointCloudXYZ::Ptr keypoints_target = estimateKeypoints(target);
    drawPointCloud(keypoints_source, keypoints_target, "Keypoints extracted");

    std::cout << "Estimating Normals.." << std::endl;
    NormalCloud::Ptr normals_source = estimateNormals(keypoints_source);
    NormalCloud::Ptr normals_target = estimateNormals(keypoints_target);

    std::cout << "Estimating Features.." << std::endl;
    //auto features_source = estimateFPFH(keypoints_source, normals_source);
    //auto features_target = estimateFPFH(keypoints_target, normals_target);
    pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> est1;

    est1.setInputCloud(keypoints_source);
    est1.setInputNormals(normals_source);
    est1.setSearchMethod(pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>));
    est1.setKSearch(100);
    est1.setSearchSurface(keypoints_source);
    FeaturePointCloud::Ptr features_source(new FeaturePointCloud);
    est1.compute(*features_source);

    pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> est2;
    est2.setInputCloud(keypoints_target);
    est2.setInputNormals(normals_target);
    est2.setSearchMethod(pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>));
    est2.setKSearch(100);
    est2.setSearchSurface(keypoints_target);
    FeaturePointCloud::Ptr features_target(new FeaturePointCloud);
    est2.compute(*features_target);

    std::cout << "finding correspondences.." << std::endl;
    // Find correspondences between keypoints in FPFH space
    pcl::CorrespondencesPtr all_correspondences(new pcl::Correspondences),
	good_correspondences(new pcl::Correspondences);
    findCorrespondences<pcl::FPFHSignature33>(features_source, features_target, *all_correspondences);
    std::cout << "all_corres:" << all_correspondences->size() << std::endl;

    // Reject correspondences based on their XYZ distance
    rejectBadCorrespondences(all_correspondences, keypoints_source, keypoints_target, *good_correspondences);

    std::cout << "good_corres:" << good_correspondences->size() << std::endl;
    for (int i = 0; i < good_correspondences->size(); i++)
	std::cerr << good_correspondences->at(i) << std::endl;
    // Obtain the best transformation between the two sets of keypoints given the remaining correspondences
    std::cout << "Obtaining transformation" << std::endl;
    pcl::registration::TransformationEstimationSVD<PointType, PointType> trans_est;
    Eigen::Matrix4f transform;
    trans_est.estimateRigidTransformation(*keypoints_source, *keypoints_target, *good_correspondences, transform);
    std::cout << transform << std::endl;

    pcl::transformPointCloud(*source, *source, transform);
    drawPointCloud(source, target, "Result after registration using features");

    PointCloudXYZ::Ptr transformed_point_cloud(new PointCloudXYZ);
    //Solution using ICP
    pcl::transformPointCloud(*source, *source, getTransform_icp(source, target));
    drawPointCloud(source, target, "ICP Result");
    return transform;
}

void matchPointClouds(PointCloudXYZ::Ptr source, PointCloudXYZ::Ptr target)
{
    auto transform = getTransform(source, target);
    pcl::transformPointCloud(*source, *source, transform);
}

PointCloudXYZ::Ptr loadPLY(std::string filename)
{
    PointCloudXYZ::Ptr cloud(new PointCloudXYZ);
    pcl::io::loadPLYFile(filename, *cloud);
    std::cout << "Size of " << filename << " " << cloud->size() << std::endl;
    return cloud;
}

#include <time.h> /* time */
void applyRandomTransform(PointCloudXYZ::Ptr cloud, float max_angle, float max_trans)
{
    srand(time(NULL));
    Eigen::Vector3f axis((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
    axis.normalize();
    float angle = (float)rand() / RAND_MAX * max_angle;
    Eigen::Vector3f translation((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
    translation *= max_trans;
    Eigen::Affine3f rotation(Eigen::AngleAxis<float>(angle, axis));
    Eigen::Affine3f transform;
    transform = Eigen::Translation3f(translation) * rotation;

    pcl::transformPointCloud(*cloud, *cloud, transform);
}

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
void normalize(PointCloudXYZ::Ptr point_cloud)
{
    Eigen::Vector4f centroid;

    std::cout << "Translating.." << std::endl;
    pcl::compute3DCentroid(*point_cloud, centroid);
    pcl::demeanPointCloud(*point_cloud, centroid, *point_cloud);
    pcl::compute3DCentroid(*point_cloud, centroid);

    pcl::PointXYZ min_point, max_point;
    pcl::getMinMax3D(*point_cloud, min_point, max_point);
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 0) = transform(0, 0) * (1 / (max_point.x - min_point.x));
    transform(1, 1) = transform(1, 1) * (1 / (max_point.y - min_point.y));
    transform(2, 2) = transform(2, 2) * (1 / (max_point.z - min_point.z));

    pcl::transformPointCloud(*point_cloud, *point_cloud, transform);

    pcl::getMinMax3D(*point_cloud, min_point, max_point);
}

int main(int argc, char** argv)
{
    std::string stageName("File loading");
    std::string mesh_path("../assets/mesh_highRes.ply");
    auto source_cloud = loadPLY(mesh_path);
    std::string cloud_path("../assets/chair-pcl-bin.ply");
    auto target_cloud = loadPLY(cloud_path);
    normalize(source_cloud);
    normalize(target_cloud);
    drawPointCloud(source_cloud, target_cloud, stageName);

    applyRandomTransform(source_cloud, 2 * M_PI, 20);
    drawPointCloud(source_cloud, target_cloud, "Result of rotation");

    matchPointClouds(source_cloud, target_cloud);
    drawPointCloud(source_cloud, target_cloud, "Final Result");

    return 0;
}
