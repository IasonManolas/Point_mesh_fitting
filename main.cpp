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
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "First cloud");

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
	viewer->spinOnce(100);
	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

////////////////////////////////////////////////////////////////////////////////
void estimateSHOT352(const PointCloudXYZ::Ptr& src,
    const PointCloudXYZ::Ptr& tgt,
    const NormalCloud::Ptr& normals_src,
    const NormalCloud::Ptr& normals_tgt,
    const PointCloudXYZ::Ptr& keypoints_src,
    const PointCloudXYZ::Ptr& keypoints_tgt,
    //pcl::PointCloud<pcl::FPFHSignature33>& fpfhs_src,
    //pcl::PointCloud<pcl::FPFHSignature33>& fpfhs_tgt)
    pcl::PointCloud<pcl::SHOT352>& shot352_src,
    pcl::PointCloud<pcl::SHOT352>& shot352_tgt)
{
    std::cout << "Computing fpfh for src" << std::endl;
    //pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
    pcl::SHOTEstimation<PointType, pcl::Normal, pcl::SHOT352> est;
    est.setInputCloud(keypoints_src);
    est.setInputNormals(normals_src);
    est.setRadiusSearch(1); // 1m
    est.setSearchSurface(src);
    est.compute(shot352_src);

    std::cout << "Computing fpfh for tgt" << std::endl;
    est.setInputCloud(keypoints_tgt);
    est.setInputNormals(normals_tgt);
    est.setSearchSurface(tgt);
    est.compute(shot352_tgt);

    std::cout << "shot352_src:" << shot352_src.size() << " "
	      << "shot352_tgt:" << shot352_tgt.size() << std::endl;
    //savePCDFile("fpfhs_tgt.pcd", out);
    // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
    //pcl_viewer fpfhs_src.pcd;
    //std::cout
    //    << "Saving pcd files" << std::endl;
    //PCLPointCloud2 s, t, out;
    //toPCLPointCloud2(*keypoints_src, s);
    //toPCLPointCloud2(fpfhs_src, t);
    //concatenateFields(s, t, out);
    //savePCDFile("fpfhs_src.pcd", out);
    //toPCLPointCloud2(*keypoints_tgt, s);
    //toPCLPointCloud2(fpfhs_tgt, t);
    //concatenateFields(s, t, out);
    //savePCDFile("fpfhs_tgt.pcd", out);
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
    rej.setMaximumDistance(1); // 1m
    rej.setInputCorrespondences(all_correspondences);
    rej.getCorrespondences(remaining_correspondences);
}

////////////////////////////////////////////////////////////////////////////////
void computeTransformation(const PointCloudXYZ::Ptr src,
    const PointCloudXYZ::Ptr tgt,
    Eigen::Matrix4f& transform)
{
    // Get an uniform grid of keypoints
    PointCloudXYZ::Ptr keypoints_src(new PointCloudXYZ),
	keypoints_tgt(new PointCloudXYZ);

    //pcl::console::print_info("Found %lu and %lu keypoints for the source and target datasets.\n", keypoints_src->points.size(), keypoints_tgt->points.size());

    // Compute normals for all points keypoint
    std::cout << "Estimating normals.." << std::endl;
    NormalCloud::Ptr normals_src(new NormalCloud),
	normals_tgt(new NormalCloud);
    //estimateNormals(src, tgt, *normals_src, *normals_tgt);
    pcl::console::print_info("Estimated %lu and %lu normals for the source and target datasets.\n", normals_src->points.size(), normals_tgt->points.size());

    std::cout << "Computing features.." << std::endl;
    // Compute FPFH features at each keypoint
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>),
	fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>);
    //    pcl::PointCloud<pcl::SHOT352>::Ptr fpfhs_src(new pcl::PointCloud<pcl::SHOT352>),
    //	fpfhs_tgt(new pcl::PointCloud<pcl::SHOT352>);
    std::cout << "Computing fpfh for src" << std::endl;
    pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> est;
    //pcl::SHOTEstimation<PointType, pcl::Normal, pcl::SHOT352> est;
    //est.setInputCloud(keypoints_src);
    est.setInputNormals(normals_src);
    est.setRadiusSearch(0.5); // 1m
    est.setSearchSurface(src);
    est.compute(*fpfhs_src);

    std::cout << "Computing fpfh for tgt" << std::endl;
    //est.setInputCloud(keypoints_tgt);
    est.setInputNormals(normals_tgt);
    est.setSearchSurface(tgt);
    est.compute(*fpfhs_tgt);

    std::cout << "fpfhs_src:" << fpfhs_src->size() << " "
	      << "fpfhs_tgt:" << fpfhs_tgt->size() << std::endl;

    std::cout << "finding correspondences.." << std::endl;
    // Find correspondences between keypoints in FPFH space
    pcl::CorrespondencesPtr all_correspondences(new pcl::Correspondences),
	good_correspondences(new pcl::Correspondences);
    findCorrespondences<pcl::FPFHSignature33>(fpfhs_src, fpfhs_tgt, *all_correspondences);
    std::cout << "all_corres:" << all_correspondences->size() << std::endl;

    // Reject correspondences based on their XYZ distance
    rejectBadCorrespondences(all_correspondences, keypoints_src, keypoints_tgt, *good_correspondences);

    std::cout << "good_corres:" << good_correspondences->size() << std::endl;
    std::cout << "Obtaining transformation" << std::endl;
    for (int i = 0; i < good_correspondences->size(); i++)
	std::cerr << good_correspondences->at(i) << std::endl;
    // Obtain the best transformation between the two sets of keypoints given the remaining correspondences
    pcl::registration::TransformationEstimationSVD<PointType, PointType> trans_est;
    trans_est.estimateRigidTransformation(*keypoints_src, *keypoints_tgt, *good_correspondences, transform);
    std::cout << transform << std::endl;

    pcl::transformPointCloud(*src, *tgt, transform);
    drawPointCloud(src, tgt, "Final Result");
}

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
void normalize(PointCloudXYZ::Ptr point_cloud)
{
    Eigen::Vector4f centroid;

    //std::cout << "Translating.." << std::endl;
    pcl::compute3DCentroid(*point_cloud, centroid);
    //std::cout << "Centroid before" << centroid.x() << " " << centroid.y() << " " << centroid.z() << std::endl;
    pcl::demeanPointCloud(*point_cloud, centroid, *point_cloud);
    pcl::compute3DCentroid(*point_cloud, centroid);
    //std::cout << "Centroid after" << centroid.x() << " " << centroid.y() << " " << centroid.z() << std::endl;

    pcl::PointXYZ min_point, max_point;
    pcl::getMinMax3D(*point_cloud, min_point, max_point);
    //std::cout << "Min before" << min_point.x << " " << min_point.y << " " << min_point.z << std::endl;
    //std::cout << "Max before" << max_point.x << " " << max_point.y << " " << max_point.z << std::endl;

    //std::cout << "Scaling.." << std::endl;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 0) = transform(0, 0) * (1 / (max_point.x - min_point.x));
    transform(1, 1) = transform(1, 1) * (1 / (max_point.y - min_point.y));
    transform(2, 2) = transform(2, 2) * (1 / (max_point.z - min_point.z));

    pcl::transformPointCloud(*point_cloud, *point_cloud, transform);

    pcl::getMinMax3D(*point_cloud, min_point, max_point);
    //std::cout << "Min after" << min_point.x << " " << min_point.y << " " << min_point.z << std::endl;
    //std::cout << "Max after" << max_point.x << " " << max_point.y << " " << max_point.z << std::endl;
}

void downsamplePointCloud(PointCloudXYZ::Ptr ptr_cloud)
{
    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    float leaf_size = 0.01;
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);

    pcl::PCLPointCloud2::Ptr ptr_cloud2(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*ptr_cloud, *ptr_cloud2);

    sor.setInputCloud(ptr_cloud2);
    sor.filter(*ptr_cloud2);

    pcl::fromPCLPointCloud2(*ptr_cloud2, *ptr_cloud);

    std::cout << "Point cloud size after filtering:" << ptr_cloud->size() << std::endl;
}

#include <pcl/registration/icp.h>
Eigen::Matrix4f getTransform_icp(PointCloudXYZ::Ptr source_cloud, PointCloudXYZ::Ptr target_cloud)
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);

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
    uniform_sampling.setRadiusSearch(0.01);
    boost::shared_ptr<PointCloudXYZ> keypoints_cloud(new PointCloudXYZ);
    uniform_sampling.filter(*keypoints_cloud);
    std::cout << "Found " << keypoints_cloud->size() << " keypoints" << std::endl;

    return keypoints_cloud;
}

NormalCloud::Ptr estimateNormals(PointCloudXYZ::ConstPtr cloud)
{
    pcl::NormalEstimation<PointType, pcl::Normal> normal_est;
    normal_est.setInputCloud(cloud);
    //normal_est.setRadiusSearch(0.06); // 50cm
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

using FeaturePointCloud = pcl::PointCloud<pcl::FPFHSignature33>;
FeaturePointCloud::Ptr estimateFPFH(PointCloudXYZ::ConstPtr cloud, NormalCloud::ConstPtr normals)
{
    std::cout << "Computing fpfh.." << std::endl;
    pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setInputCloud(cloud);
    fpfh_est.setInputNormals(normals);
    fpfh_est.setSearchMethod(pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>));
    fpfh_est.setKSearch(40);
    fpfh_est.setSearchSurface(cloud);
    //fpfh_est.setRadiusSearch(0.1);

    FeaturePointCloud::Ptr features(new FeaturePointCloud);
    fpfh_est.compute(*features);

    std::cout << "features:" << features->size() << std::endl;

    return features;
}

Eigen::Matrix4f getTransform_features(PointCloudXYZ::Ptr source, PointCloudXYZ::Ptr target)
{
    PointCloudXYZ::Ptr keypoints_source = estimateKeypoints(source);
    PointCloudXYZ::Ptr keypoints_target = estimateKeypoints(target);
    drawPointCloud(keypoints_source, keypoints_target, "Keypoints extracted");

    NormalCloud::Ptr normals_source = estimateNormals(keypoints_source);
    NormalCloud::Ptr normals_target = estimateNormals(keypoints_target);

    auto features_source = estimateFPFH(keypoints_source, normals_source);
    auto features_target = estimateFPFH(keypoints_target, normals_target);

    return Eigen::Matrix4f::Identity();
}

int main(int argc, char** argv)
{
    std::string stageName("File loading");
    //Load Mesh
    std::cout << "Loading OBJ.." << std::endl;
    std::string mesh_path("../assets/mesh_highRes.ply");
    PointCloudXYZ::Ptr source_cloud(new PointCloudXYZ);
    pcl::io::loadPLYFile(mesh_path, *source_cloud);
    std::cout << "Size of mesh:" << source_cloud->size() << std::endl;
    //Load Point Cloud
    std::cout << "Loading cloud.." << std::endl;
    std::string cloud_path("../assets/chair-pcl-bin.ply");
    PointCloudXYZ::Ptr target_cloud(new PointCloudXYZ);
    pcl::io::loadPLYFile(cloud_path, *target_cloud);
    std::cout << "Size of point cloud:" << target_cloud->size() << std::endl;

    //normalize(source_cloud);
    //normalize(target_cloud);

    drawPointCloud(source_cloud, target_cloud, stageName);

    PointCloudXYZ::Ptr transformed_point_cloud(new PointCloudXYZ);
    //Solution using ICP
    //Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

    //float theta = M_PI / 14; // The angle of rotation in radians
    //transform_1(0, 0) = cos(theta);
    //transform_1(0, 1) = -sin(theta);
    //transform_1(1, 0) = sin(theta);
    //transform_1(1, 1) = cos(theta);
    //pcl::transformPointCloud(*source_cloud, *source_cloud, transform_1);
    //drawPointCloud(source_cloud, target_cloud, "Result of rotation");
    //pcl::transformPointCloud(*source_cloud, *transformed_point_cloud, getTransform_icp(source_cloud, target_cloud));
    //drawPointCloud(transformed_point_cloud, target_cloud, "ICP Result");

    //Solution using features
    auto transform = getTransform_features(source_cloud, target_cloud);
    pcl::transformPointCloud(*source_cloud, *transformed_point_cloud, transform);
    return 0;
}
