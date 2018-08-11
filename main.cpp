#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/point_types.h>

#include <pcl/conversions.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/cloud_viewer.h>

using PointType = pcl::PointXYZ;
using PointCloudXYZ = pcl::PointCloud<PointType>;
using NormalCloud = pcl::PointCloud<pcl::Normal>;
PointCloudXYZ::Ptr src, tgt;

void drawPointCloud(std::vector<PointCloudXYZ::Ptr> cloud_ptrs)
{
    pcl::visualization::CloudViewer viewer("PCL Viewer");

    for (size_t i = 0; i < cloud_ptrs.size(); i++) {
	std::cout << "Cloud size:" << cloud_ptrs[i]->size() << std::endl;

	viewer.showCloud(cloud_ptrs[i], std::to_string(i));
    }
    while (!viewer.wasStopped()) {
    }
}
////////////////////////////////////////////////////////////////////////////////
void estimateKeypoints(const PointCloudXYZ::Ptr& src,
    const PointCloudXYZ::Ptr& tgt,
    PointCloudXYZ& keypoints_src,
    PointCloudXYZ& keypoints_tgt)
{
    // Get an uniform grid of keypoints
    pcl::UniformSampling<PointType> uniform;
    uniform.setRadiusSearch(10); // 1m

    uniform.setInputCloud(src);
    uniform.filter(keypoints_src);

    uniform.setInputCloud(tgt);
    uniform.filter(keypoints_tgt);

    // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
    // pcl_viewer source_pcd keypoints_src.pcd -ps 1 -ps 10
    //pcl::io::savePCDFileBinary("keypoints_src.pcd", keypoints_src);
    //pcl::io::savePCDFileBinary("keypoints_tgt.pcd", keypoints_tgt);
}

////////////////////////////////////////////////////////////////////////////////
void estimateNormals(const PointCloudXYZ::Ptr& src,
    const PointCloudXYZ::Ptr& tgt,
    NormalCloud& normals_src,
    NormalCloud& normals_tgt)
{
    pcl::NormalEstimation<PointType, pcl::Normal> normal_est;
    normal_est.setInputCloud(src);
    normal_est.setRadiusSearch(0.1); // 50cm
    normal_est.compute(normals_src);

    normal_est.setInputCloud(tgt);
    normal_est.compute(normals_tgt);

    // For debugging purposes only: uncomment the lines below and use pcl_viewer to view the results, i.e.:
    // pcl_viewer normals_src.pcd
    pcl::PointCloud<pcl::PointNormal> s, t;
    pcl::copyPointCloud<PointType, pcl::PointNormal>(*src, s);
    pcl::copyPointCloud<pcl::Normal, pcl::PointNormal>(normals_src, s);
    pcl::copyPointCloud<PointType, pcl::PointNormal>(*tgt, t);
    pcl::copyPointCloud<pcl::Normal, pcl::PointNormal>(normals_tgt, t);
    //savePCDFileBinary("normals_src.pcd", s);
    //savePCDFileBinary("normals_tgt.pcd", t);
}

////////////////////////////////////////////////////////////////////////////////
void estimateFPFH(const PointCloudXYZ::Ptr& src,
    const PointCloudXYZ::Ptr& tgt,
    const NormalCloud::Ptr& normals_src,
    const NormalCloud::Ptr& normals_tgt,
    const PointCloudXYZ::Ptr& keypoints_src,
    const PointCloudXYZ::Ptr& keypoints_tgt,
    pcl::PointCloud<pcl::FPFHSignature33>& fpfhs_src,
    pcl::PointCloud<pcl::FPFHSignature33>& fpfhs_tgt)
{
    std::cout << "Computing fpfh for src" << std::endl;
    pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setInputCloud(keypoints_src);
    fpfh_est.setInputNormals(normals_src);
    fpfh_est.setRadiusSearch(10); // 1m
    fpfh_est.setSearchSurface(src);
    fpfh_est.compute(fpfhs_src);

    std::cout << "Computing fpfh for tgt" << std::endl;
    fpfh_est.setInputCloud(keypoints_tgt);
    fpfh_est.setInputNormals(normals_tgt);
    fpfh_est.setSearchSurface(tgt);
    fpfh_est.compute(fpfhs_tgt);

    std::cout << "fpfs_src:" << fpfhs_src.size() << " "
	      << "fpfhs_tgt:" << fpfhs_tgt.size() << std::endl;
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

////////////////////////////////////////////////////////////////////////////////
void findCorrespondences(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs_src,
    const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfhs_tgt,
    pcl::Correspondences& all_correspondences)
{
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
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
void computeTransformation(const PointCloudXYZ::Ptr& src,
    const PointCloudXYZ::Ptr& tgt,
    Eigen::Matrix4f& transform)
{
    // Get an uniform grid of keypoints
    PointCloudXYZ::Ptr keypoints_src(new PointCloudXYZ),
	keypoints_tgt(new PointCloudXYZ);

    estimateKeypoints(src, tgt, *keypoints_src, *keypoints_tgt);
    pcl::console::print_info("Found %lu and %lu keypoints for the source and target datasets.\n", keypoints_src->points.size(), keypoints_tgt->points.size());

    // Compute normals for all points keypoint
    std::cout << "Estimating normals.." << std::endl;
    NormalCloud::Ptr normals_src(new NormalCloud),
	normals_tgt(new NormalCloud);
    estimateNormals(src, tgt, *normals_src, *normals_tgt);
    pcl::console::print_info("Estimated %lu and %lu normals for the source and target datasets.\n", normals_src->points.size(), normals_tgt->points.size());

    std::cout << "Computing features.." << std::endl;
    // Compute FPFH features at each keypoint
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>),
	fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>);
    //estimateFPFH(src, tgt, normals_src, normals_tgt, keypoints_src, keypoints_tgt, *fpfhs_src, *fpfhs_tgt);
    std::cout << "Computing fpfh for src" << std::endl;
    pcl::FPFHEstimation<PointType, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setInputCloud(src);
    fpfh_est.setInputNormals(normals_src);
    fpfh_est.setRadiusSearch(1); // 1m
    fpfh_est.setSearchSurface(src);
    fpfh_est.compute(*fpfhs_src);

    std::cout << "Computing fpfh for tgt" << std::endl;
    fpfh_est.setInputCloud(tgt);
    fpfh_est.setInputNormals(normals_tgt);
    fpfh_est.setSearchSurface(tgt);
    fpfh_est.compute(*fpfhs_tgt);

    std::cout << "fpfs_src:" << fpfhs_src->size() << " "
	      << "fpfhs_tgt:" << fpfhs_tgt->size() << std::endl;

    std::cout << "finding correspondences.." << std::endl;
    // Find correspondences between keypoints in FPFH space
    pcl::CorrespondencesPtr all_correspondences(new pcl::Correspondences),
	good_correspondences(new pcl::Correspondences);
    findCorrespondences(fpfhs_src, fpfhs_tgt, *all_correspondences);
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

    pcl::transformPointCloud(*src, *src, transform);
    drawPointCloud(std::vector<PointCloudXYZ::Ptr>{ src, tgt });
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
/* ---[ */
int main(int argc, char** argv)
{
    //Load Mesh
    std::cout << "Loading OBJ.." << std::endl;
    pcl::PLYReader obj_reader;
    std::string obj_path("../assets/mesh.ply");
    PointCloudXYZ::Ptr mesh_cloud(new PointCloudXYZ);
    obj_reader.read(obj_path, *mesh_cloud);
    //Load Point Cloud
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2());
    pcl::PLYReader reader;
    std::string ply_path("../assets/chair-pcl-bin.ply");
    reader.read(ply_path, *cloud2);
    PointCloudXYZ::Ptr cloud(new PointCloudXYZ);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);
    std::cout << "Size of point cloud:" << cloud->size() << std::endl;
    //Normalize
    normalize(mesh_cloud);
    normalize(cloud);
    pcl::toPCLPointCloud2(*cloud, *cloud2);
    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(cloud2);
    float leaf_size = 0.01;
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    //pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());
    sor.filter(*cloud2);
    //sor.filter(*mesh_cloud);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);
    std::cerr << "PointCloud after filtering: " << cloud->width * cloud->height
	      << " data points (" << pcl::getFieldsList(*cloud) << ").";

    //drawPointCloud(std::vector<PointCloudXYZ::Ptr>{ cloud, mesh_cloud });

    // Compute the best transformtion
    Eigen::Matrix4f transform;
    computeTransformation(cloud, mesh_cloud, transform);
    std::cout << "Computed Transformation.." << std::endl;

    // Transform the data and write it to disk
    //PointCloudXYZ output;
    //transformPointCloud(src, output, transform);
    //savePCDFileBinary("source_transformed.pcd", output);
    return 0;
}
/* ]--- */
