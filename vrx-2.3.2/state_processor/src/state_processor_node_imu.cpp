#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "cluster_msg/msg/point_cloud_cluster.hpp"
#include "state_msg/msg/state.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <Eigen/Dense>
#include <unordered_set>

const float gravity_acc = 9.81;

class StateProcessorNode : public rclcpp::Node {
public:
    StateProcessorNode(const std::string& robot_name) : Node("state_processor_node_" + robot_name) {
        imu_subscriber_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Imu>>(
            this, "/"+robot_name+"/sensors/imu/imu/data");
        lidar_subscriber_ = std::make_shared<message_filters::Subscriber<cluster_msg::msg::PointCloudCluster>>(
            this, "/"+robot_name+"/sensors/lidars/lidar_wamv_sensor/clusters");

        // Synchronize IMU and LiDAR messages based on timestamps
        time_sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Imu, cluster_msg::msg::PointCloudCluster>>(*imu_subscriber_, *lidar_subscriber_, 10);
        time_sync_->registerCallback(&StateProcessorNode::synchronizeCallback, this);

        // Publisher for synchronized timestamps
        // timestamp_publisher_ = this->create_publisher<std_msgs::msg::String>("/"+robot_name+"/sync_timestamp", 10);

        // State publisher
        state_publisher_ = this->create_publisher<state_msg::msg::State>("/"+robot_name+"/robot_state", 10);

        if(robot_name == "wamv1") print_latency = true;
        else print_latency = false;

        last_imu_msg = nullptr;
        last_lidar_msg = nullptr;

        // In simulation an acceleration component pointing upward with magnitute the same as gravity acceleration is added to IMU measurement
        gravity = Eigen::Vector3f(0.0, 0.0, gravity_acc);

        // Initialize clock
        lastMessageTime_ = std::chrono::steady_clock::now();

        // Initialize robot velocity
        robotVelocity.linear.x = 0.0;
        robotVelocity.linear.y = 0.0;
        robotVelocity.linear.z = 0.0;
        robotVelocity.angular.x = 0.0;
        robotVelocity.angular.y = 0.0;
        robotVelocity.angular.z = 0.0;

        // Initialize thresholds
        max_robot_speed = 6.0;
        max_num_points_change = 0.3;
    }

private:
    void synchronizeCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg, const cluster_msg::msg::PointCloudCluster::SharedPtr lidar_msg) {
        // Your synchronization logic goes here
        // Access imu_msg and lidar_msg data for fusion or processing

        auto now = std::chrono::steady_clock::now();
        
        geometry_msgs::msg::Point velocity_zero;
        velocity_zero.x = 0.0;
        velocity_zero.y = 0.0;
        velocity_zero.z = 0.0;
        std::vector<geometry_msgs::msg::Point> cluster_velocities(lidar_msg->cluster_centroids.size(),velocity_zero);

        if(last_lidar_msg != nullptr){
            // Timestamp latency (ms)
            float last_timestamp = last_lidar_msg->header.stamp.sec + last_lidar_msg->header.stamp.nanosec * 1e-9;
            float curr_timestamp = lidar_msg->header.stamp.sec + lidar_msg->header.stamp.nanosec * 1e-9;
            float diff_timestamp = curr_timestamp - last_timestamp;

            // Clock time latency (ms)
            auto clock_diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastMessageTime_).count();
            
            // Subtract gravity acceleration from IMU measurement
            subtractGravityAcceleration();

            if(print_latency){
                // RCLCPP_INFO(this->get_logger(), "Timestamp accum time: %f ms, Exact accum time: %ld ms", trunc(diff_timestamp * 1e3), clock_diff);
                RCLCPP_INFO(this->get_logger(), "IMU acc x: %f, y: %f, z: %f", last_imu_msg->linear_acceleration.x,
                            last_imu_msg->linear_acceleration.y, last_imu_msg->linear_acceleration.z);
                RCLCPP_INFO(this->get_logger(), "Velocity x: %f, y: %f, z: %f", robotVelocity.linear.x,
                            robotVelocity.linear.y, robotVelocity.linear.z);
            }

            // Compute pose translation and rotation from last timestamp (Reversed to project points)
            Eigen::Vector3f reverse_translation(-1.0 * robotVelocity.linear.x * diff_timestamp,
                                                -1.0 * robotVelocity.linear.y * diff_timestamp,
                                                -1.0 * robotVelocity.linear.z * diff_timestamp);
            
            float roll = -1.0 * robotVelocity.angular.x * diff_timestamp;
            float pitch = -1.0 * robotVelocity.angular.y * diff_timestamp;
            float yaw = -1.0 * robotVelocity.angular.z * diff_timestamp;

            Eigen::Quaternionf reverse_quaternion = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
                                                     Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
                                                     Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());

            Eigen::Matrix3f reverse_rotation = reverse_quaternion.toRotationMatrix();

            // Update current robot velocity
            robotVelocity.linear.x += last_imu_msg->linear_acceleration.x * diff_timestamp;
            robotVelocity.linear.y += last_imu_msg->linear_acceleration.y * diff_timestamp;
            robotVelocity.linear.z += last_imu_msg->linear_acceleration.z * diff_timestamp;
            robotVelocity.angular.x = imu_msg->angular_velocity.x;
            robotVelocity.angular.y = imu_msg->angular_velocity.y;
            robotVelocity.angular.z = imu_msg->angular_velocity.z;

            computeClusterVelocities(lidar_msg,reverse_translation,reverse_rotation,cluster_velocities,diff_timestamp);
        }

        // Create state message and publish
        state_msg::msg::State state;
        state.header = lidar_msg->header;
        
        geometry_msgs::msg::Point self_velocity;
        self_velocity.x = robotVelocity.linear.x;
        self_velocity.y = robotVelocity.linear.y;
        self_velocity.z = robotVelocity.linear.z;
        state.self_velocity = self_velocity;
        
        state.object_positions = lidar_msg->cluster_centroids;
        state.object_radii = lidar_msg->cluster_radii;
        state.object_velocities = cluster_velocities;

        last_imu_msg = imu_msg;
        last_lidar_msg = lidar_msg;
        lastMessageTime_ = now;

        state_publisher_->publish(state);

        // Publish synchronized timestamps
        // auto sync_timestamps = std_msgs::msg::String();
        // sync_timestamps.data = "IMU timestamp: " + std::to_string(imu_msg->header.stamp.sec) + "." + std::to_string(imu_msg->header.stamp.nanosec) +
        //                        ", LiDAR timestamp: " + std::to_string(lidar_msg->header.stamp.sec) + "." + std::to_string(lidar_msg->header.stamp.nanosec);
        // timestamp_publisher_->publish(sync_timestamps);
    }

    void subtractGravityAcceleration(){
        Eigen::Quaternionf imu_orientation(last_imu_msg->orientation.w,last_imu_msg->orientation.x,
                                           last_imu_msg->orientation.y,last_imu_msg->orientation.z);

        Eigen::Vector3f projected_gravity = imu_orientation.conjugate().toRotationMatrix() * gravity;

        last_imu_msg->linear_acceleration.x -= projected_gravity(0);
        last_imu_msg->linear_acceleration.y -= projected_gravity(1);
        last_imu_msg->linear_acceleration.z -= projected_gravity(2);
    }

    void computeClusterVelocities(const cluster_msg::msg::PointCloudCluster::SharedPtr lidar_msg,
                                  const Eigen::Vector3f& project_translation, const Eigen::Matrix3f& project_rotation,
                                  std::vector<geometry_msgs::msg::Point>& cluster_velocities, float diff_time){
        
        if(last_lidar_msg->cluster_centroids.size() == 0 || lidar_msg->cluster_centroids.size() == 0) return;

        // Project clusters in the last frame to the current frame
        for(int i=0; i<last_lidar_msg->cluster_centroids.size(); i++){
            Eigen::Vector3f centroid(last_lidar_msg->cluster_centroids[i].x,
                                     last_lidar_msg->cluster_centroids[i].y,
                                     last_lidar_msg->cluster_centroids[i].z);
            
            centroid = project_rotation * (centroid + project_translation);

            last_lidar_msg->cluster_centroids[i].x = centroid(0);
            last_lidar_msg->cluster_centroids[i].y = centroid(1);
            last_lidar_msg->cluster_centroids[i].z = centroid(2);
        }

        std::unordered_set<int> paired_clusters;
        for(int i=0; i<lidar_msg->cluster_centroids.size(); i++){
            
            int min_cluster_id = 0;
            float min_dist = computeDistance(lidar_msg->cluster_centroids[i],last_lidar_msg->cluster_centroids[0]);
            
            for(int j=1; j<last_lidar_msg->cluster_centroids.size(); j++){
                float dist = computeDistance(lidar_msg->cluster_centroids[i],last_lidar_msg->cluster_centroids[j]);
                if(dist < min_dist){
                    min_dist = dist;
                    min_cluster_id = j;
                }
            }

            if(paired_clusters.find(min_cluster_id) != paired_clusters.end()) continue;

            // Distance is too large
            if(min_dist > max_robot_speed * diff_time) continue;

            // The difference in number of points is too large
            if(!compareNumPoints(lidar_msg->cluster_point_counts[i],last_lidar_msg->cluster_point_counts[min_cluster_id])) continue;

            paired_clusters.insert(min_cluster_id);

            cluster_velocities[i].x = (lidar_msg->cluster_centroids[i].x - last_lidar_msg->cluster_centroids[min_cluster_id].x) / diff_time;
            cluster_velocities[i].y = (lidar_msg->cluster_centroids[i].y - last_lidar_msg->cluster_centroids[min_cluster_id].y) / diff_time;
            cluster_velocities[i].z = (lidar_msg->cluster_centroids[i].z - last_lidar_msg->cluster_centroids[min_cluster_id].z) / diff_time;
        }
    }

    float computeDistance(geometry_msgs::msg::Point& p1, geometry_msgs::msg::Point& p2){
        // Compute distance in x-y plane
        return sqrt((p1.x - p2.x)*(p1.x - p2.x)+(p1.y - p2.y)*(p1.y - p2.y));
    }

    bool compareNumPoints(uint32_t num1, uint32_t num2){
        uint32_t num_max = std::max(num1,num2);
        uint32_t num_min = std::min(num1,num2);
        float num_diff = num_max - num_min;
        float ratio_diff = num_diff / num_max;
        return ratio_diff <= max_num_points_change;
    }

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Imu>> imu_subscriber_;
    std::shared_ptr<message_filters::Subscriber<cluster_msg::msg::PointCloudCluster>> lidar_subscriber_;
    std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Imu, cluster_msg::msg::PointCloudCluster>> time_sync_;
    // rclcpp::Publisher<std_msgs::msg::String>::SharedPtr timestamp_publisher_;
    rclcpp::Publisher<state_msg::msg::State>::SharedPtr state_publisher_;
    
    bool print_latency;
    sensor_msgs::msg::Imu::SharedPtr last_imu_msg;
    cluster_msg::msg::PointCloudCluster::SharedPtr last_lidar_msg;
    std::chrono::time_point<std::chrono::steady_clock> lastMessageTime_; // clock time of last message

    Eigen::Vector3f gravity;

    geometry_msgs::msg::Twist robotVelocity;

    float max_robot_speed;
    float max_num_points_change; // Threshold of change in number of points of the same cluster between consecutive frames (ratio)  
};

int main(int argc, char** argv) {
    if(argc < 2){
        fprintf(stderr,"Usage: %s robot_name\n",argv[0]);
        return 1;
    }

    rclcpp::init(argc, argv);
    auto node = std::make_shared<StateProcessorNode>(std::string(argv[1]));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
