#include <rclcpp/rclcpp.hpp>
#include <rcpputils/endian.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensing_msgs/msg/encoded_image.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <iostream>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <filesystem>
#include <cassert>
#include <algorithm>
#include <cstddef>
#include <type_traits>


using std::placeholders::_1;


template <typename T>
T deserialize(std::vector<unsigned char>& vec)
{
    static_assert(std::is_trivially_copyable<T>::value, "Deserialization requires trivially copyable types");

    if (sizeof(T) > vec.size())
    {
        throw std::out_of_range("Not enough bytes in vector to deserialize the requested type");
    }

    T value = 0;
    const size_t typeSize = sizeof(T);

    for (size_t i = 0; i < typeSize; ++i)
    {
        value |= static_cast<T>(vec[i]) << ((typeSize - 1 - i) * 8);
    }

    // Erase the consumed bytes from the vector
    vec.erase(vec.begin(), vec.begin() + typeSize);

    return value;
}


class PredictPose : public rclcpp::Node
{
public:
    PredictPose()
        : Node("predict_pose")
    {
        declare_parameter("model_msg_file", "");
        declare_parameter("model_post_file", "");
        declare_parameter("cam_namespace_other", "");
        declare_parameter("pose_topic_name", "rel_pos");

        std::string model_msg_file;
        std::string model_post_file;
        std::string pose_topic_name;
        get_parameter("model_msg_file", model_msg_file);
        get_parameter("model_post_file", model_post_file);
        get_parameter("cam_namespace_other", cam_namespace_other_);
        get_parameter("pose_topic_name", pose_topic_name);

        assert(!cam_namespace_other_.empty());
        assert(!(cam_namespace_other_.back() == '/'));

        auto pkg_path = std::filesystem::path(ament_index_cpp::get_package_share_directory("sensing_cpp"));

        auto model_msg_path = pkg_path / model_msg_file;
        model_version_ = model_version_from_path(model_msg_path);
        RCLCPP_INFO(get_logger(), "Loading msg model version %s from %s", model_version_.c_str(), model_msg_path.c_str());
        model_msg_ = torch::jit::load(model_msg_path);
        model_msg_.eval();
        if (model_msg_file.find("cpu") != std::string::npos)
        {
            model_msg_.to(torch::kCPU);
            // RCLCPP_INFO(get_logger(), "Msg model loaded on CPU");
        }
        else
        {
            model_msg_.to(torch::kCUDA);
            // RCLCPP_INFO(get_logger(), "Msg model loaded on GPU");
        }


        auto model_post_path = pkg_path / model_post_file;
        assert(model_version_ == model_version_from_path(model_post_path));
        RCLCPP_INFO(get_logger(), "Loading post model from %s", model_post_path.c_str());
        model_post_ = torch::jit::load(model_post_path);
        model_post_.eval();
        if (model_post_file.find("cpu") != std::string::npos)
        {
            model_post_.to(torch::kCPU);
            // RCLCPP_INFO(get_logger(), "Post model loaded on CPU");
        }
        else
        {
            model_post_.to(torch::kCUDA);
            // RCLCPP_INFO(get_logger(), "Post model loaded on GPU");
        }

        std::ostringstream cam_topic_other;
        cam_topic_other << cam_namespace_other_ << "/enc";

        enc_sub_self_ =
            create_subscription<sensing_msgs::msg::EncodedImage>(
                "enc",
                rclcpp::SensorDataQoS(),
                std::bind(&PredictPose::enc_self_callback, this, _1)
            );

        enc_sub_other_ =
            create_subscription<sensing_msgs::msg::EncodedImage>(
                cam_topic_other.str(),
                rclcpp::SensorDataQoS(),
                std::bind(&PredictPose::enc_other_callback, this, _1)
            );

        pose_publisher_ =
            create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
                pose_topic_name,
                rclcpp::SensorDataQoS()
            );

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(get_logger(), "Initialized");
    }

    ~PredictPose()
    {
        RCLCPP_INFO(get_logger(), "Destroyed");
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<sensing_msgs::msg::EncodedImage>::SharedPtr enc_sub_self_;
    rclcpp::Subscription<sensing_msgs::msg::EncodedImage>::SharedPtr enc_sub_other_;

    torch::jit::Module model_msg_;
    torch::jit::Module model_post_;
    std::string model_version_;
    std::string cam_namespace_other_;

    torch::Tensor other_enc_;;

    std::string model_version_from_path(std::filesystem::path& model_path)
    {
        auto model_stem = model_path.stem().string();
        size_t pos = model_stem.find("_");
        return model_stem.substr(0, pos);
    }

    void enc_self_callback(const sensing_msgs::msg::EncodedImage::ConstSharedPtr& enc)
    {
        if (other_enc_.numel() == 0)
        {
            RCLCPP_INFO(get_logger(), "Wait for other enc...");
            return;
        }

        float dt_rx = (get_clock()->now() - enc->img_stamp).nanoseconds() / 1e9;
        torch::Tensor self_enc = msg_to_tensor(enc);

        // TODO, check if self_enc is on CPU or GPU
        auto out = model_msg_.forward({self_enc.to(torch::kCUDA), other_enc_});
        // auto out = model_msg_.forward({self_enc.to(torch::kCPU), other_enc_});
        auto pred = model_post_.forward({out.toTensor()}).toTuple()->elements();
        auto pos = pred[0].toTensor().squeeze(0).to(torch::kCPU);
        auto pos_var = pred[1].toTensor().squeeze(0).to(torch::kCPU);
        auto rot = pred[2].toTensor().squeeze(0).to(torch::kCPU);
        auto rot_var = pred[3].toTensor().squeeze(0).to(torch::kCPU);

        geometry_msgs::msg::PoseWithCovarianceStamped msg;
        msg.header.stamp = get_clock()->now();
        msg.pose.pose.position.x = pos[0].item<double>();
        msg.pose.pose.position.y = pos[1].item<double>();
        msg.pose.pose.position.z = pos[2].item<double>();
        // Convert quaternion to Euler angles (roll, pitch, yaw)
        msg.pose.pose.orientation.x = rot[0].item<double>();
        msg.pose.pose.orientation.y = rot[1].item<double>();
        msg.pose.pose.orientation.z = rot[2].item<double>();
        msg.pose.pose.orientation.w = rot[3].item<double>();

        tf2::Quaternion q(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        );

        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        roll = roll * 180.0 / M_PI;
        pitch = pitch * 180.0 / M_PI;
        yaw = yaw * 180.0 / M_PI;

        RCLCPP_INFO(get_logger(), "Orientation (roll, pitch, yaw) in degrees: %f, %f, %f", roll, pitch, yaw);

        auto cov = torch::diag(torch::cat({pos_var, rot_var.repeat(3)}, 0));
        auto cov_flat = cov.flatten().to(torch::kFloat64).contiguous();
        auto data_start = cov_flat.data_ptr<double>();
        auto el_size = torch::elementSize(torch::typeMetaToScalarType(cov_flat.dtype()));
        assert(el_size == sizeof(double));
        size_t data_len = cov_flat.numel() * el_size;
        std::copy(data_start, data_start + data_len, msg.pose.covariance.begin());

        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->get_clock()->now();
        t.header.frame_id = get_namespace();
        t.child_frame_id = cam_namespace_other_;

        t.transform.translation.x = msg.pose.pose.position.x;
        t.transform.translation.y = msg.pose.pose.position.y;
        t.transform.translation.z = msg.pose.pose.position.z;
        t.transform.rotation.x = msg.pose.pose.orientation.x;
        t.transform.rotation.y = msg.pose.pose.orientation.y;
        t.transform.rotation.z = msg.pose.pose.orientation.z;
        t.transform.rotation.w = msg.pose.pose.orientation.w;

        float dt_proc = (get_clock()->now() - enc->img_stamp).nanoseconds() / 1e9;

        pose_publisher_->publish(std::move(msg));
        tf_broadcaster_->sendTransform(t);

        RCLCPP_INFO(get_logger(), "Processed pose dt rx %f proc %f", dt_rx, dt_proc);
    }

    void enc_other_callback(const sensing_msgs::msg::EncodedImage::ConstSharedPtr& enc)
    {
        float dt_rx = (get_clock()->now() - enc->img_stamp).nanoseconds() / 1e9;

        torch::Tensor t = msg_to_tensor(enc);
        // TODO, check if other_enc_ is on CPU or GPU
        other_enc_ = t.to(torch::kCUDA);
        // other_enc_ = t.to(torch::kCPU);

        float dt_proc = (get_clock()->now() - enc->img_stamp).nanoseconds() / 1e9;

        RCLCPP_INFO(get_logger(), "Received other dt rx %f proc %f", dt_rx, dt_proc);
    }



    torch::Tensor msg_to_tensor(const sensing_msgs::msg::EncodedImage::ConstSharedPtr& msg)
    {
        assert((msg->dtype == "float16") || (msg->dtype == "float32"));
        auto dtype = msg->dtype == "float16" ? torch::kFloat16 : torch::kFloat32;
        auto options = torch::TensorOptions().dtype(dtype);
        size_t expected_raw_size = torch::elementSize(dtype) * msg->patches * msg->features;
        assert(msg->data.size() == expected_raw_size); //, "Unexpected message data size");
        return torch::from_blob(const_cast<unsigned char*>(msg->data.data()), {1, msg->patches, msg->features}, options);
    }
};

static std::shared_ptr<PredictPose> pose_node = nullptr;


int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    pose_node = std::make_shared<PredictPose>();
    rclcpp::spin(pose_node);
    rclcpp::shutdown();
    return 0;
}
