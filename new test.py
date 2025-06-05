import numpy as np
import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, NavSatFix, Bool
from std_msgs.msg import Float32
from mechaship_interfaces.msg import RgbwLedColor
from tf_transformations import euler_from_quaternion
from cv_bridge import CvBridge
from geopy.distance import geodesic
from statistics import median
from math import degrees
import os
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

def vincenty_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def set_risk_zone(array, center, spread):
    array[center] = 1
    for i in range(1, spread + 1):
        if center - i >= 0:
            array[center - i] = 1
        if center + i <= 180:
            array[center + i] = 1
    return array

class NavigationNode(Node):
    def __init__(self):
        super().__init__("Auto_sailing")
        self.imu_heading = 90.0
        self.max_risk_threshold = 60.0
        self.key_target_degree = 90.0
        self.target_imu_angle = 90.0
        self.median_latitude = 35.0
        self.median_longitude = 126.0
        self.latitude_buffer = []
        self.longitude_buffer = []
        self.thruster_msg = 0.0

        self.br = CvBridge()
        self.camera_subscription = None

        self.color = RgbwLedColor()
        self.color.white = 20

        # [경유점 리스트] 예시: [(lat1, lon1), (lat2, lon2), ...]
        self.waypoints = [
            (35.0, 126.0),
            (35.05, 126.05),
            (35.1, 126.1)
        ]
        self.current_wp_idx = 0
        self.arrival_threshold = 5.0  # [m] 경유점 도달반경

        self.sailing_section = "navigate"

        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        gps_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        imu_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.lidar_subscription = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, lidar_qos
        )
        self.create_subscription(Imu, "/imu", self.imu_callback, imu_qos)
        self.create_subscription(NavSatFix, "/gps/fix", self.gps_callback, gps_qos)
        self.create_subscription(Bool, "/sensor/emo/status", self.emo_switch_callback, 10)

        self.rgbw_led_publisher = self.create_publisher(RgbwLedColor, "/actuator/rgbwled/color", 10)
        self.key_publisher = self.create_publisher(Float32, "/actuator/key/degree", 10)
        self.thruster_publisher = self.create_publisher(Float32, "/actuator/thruster/percentage", 10)

        self.create_timer(0.5, self.median_gps)
        self.create_timer(1.0, self.timer_callback)

    def emo_switch_callback(self, data: Bool):
        if data.data:
            self.all_stop()

    def all_stop(self):
        self.key_publisher.publish(Float32(data=90.0))
        self.thruster_publisher.publish(Float32(data=0.0))
        os.system("killall -SIGINT ros2")

    def timer_callback(self):
        if self.sailing_section == "end":
            self.thruster_msg = 0.0
            self.key_target_degree = 90.0
            self.color.white = 0
            self.color.green = 0
            self.color.red = 0
            self.color.blue = 0
        else:
            self.thruster_msg = 20.0

        self.key_publisher.publish(Float32(data=float(self.key_target_degree)))
        self.thruster_publisher.publish(Float32(data=float(self.thruster_msg)))
        self.rgbw_led_publisher.publish(self.color)

    def gps_callback(self, msg: NavSatFix):
        self.latitude_buffer.append(msg.latitude)
        self.longitude_buffer.append(msg.longitude)

    def median_gps(self):
        if self.latitude_buffer and self.longitude_buffer:
            self.median_latitude = median(self.latitude_buffer)
            self.median_longitude = median(self.longitude_buffer)
            self.get_logger().info(f"GPS: latitude={self.median_latitude}, longitude={self.median_longitude}")
            self.latitude_buffer.clear()
            self.longitude_buffer.clear()

        # ---------- Waypoint 추종 ----------
        if self.sailing_section == "navigate" and self.current_wp_idx < len(self.waypoints):
            target_lat, target_lon = self.waypoints[self.current_wp_idx]
            dist = vincenty_distance(self.median_latitude, self.median_longitude, target_lat, target_lon)
            bearing = self.calculate_bearing(self.median_latitude, self.median_longitude, target_lat, target_lon)

            self.get_logger().info(
                f"[WP{self.current_wp_idx+1}/{len(self.waypoints)}] 거리={dist:.1f}m, 방위각={bearing:.1f}deg"
            )

            self.target_imu_angle = bearing  # 목표 방위각 갱신

            if dist < self.arrival_threshold:
                self.get_logger().info(f"경유점 {self.current_wp_idx+1} 도달!")
                self.current_wp_idx += 1
                if self.current_wp_idx >= len(self.waypoints):
                    self.get_logger().info("모든 경유점 도달! 정지합니다.")
                    self.sailing_section = "end"

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        # 위경도(도) → 라디안 변환
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        return bearing

    def imu_callback(self, msg: Imu):
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        roll_rad, pitch_rad, yaw_rad = euler_from_quaternion(quaternion)
        yaw = degrees(yaw_rad)
        if yaw <= 90.0:
            yaw_degree = 90.0 - yaw
        elif yaw > 90.0:
            yaw_degree = 450.0 - yaw

        self.imu_heading = yaw_degree

    def lidar_callback(self, data):
        ranges = np.array(data.ranges)
        relevant_data = ranges[500:1500]
        relevant_data = relevant_data[(relevant_data != 0) & (relevant_data != float("inf"))]

        cumulative_distance = np.zeros(181)
        sample_count = np.zeros(181)
        average_distance = np.zeros(181)
        risk_values = np.zeros(181)
        risk_map = np.zeros(181)

        for i in range(len(relevant_data)):
            length = relevant_data[i]
            angle_index = round((len(relevant_data) - 1 - i) * 180 / len(relevant_data))
            cumulative_distance[angle_index] += length
            sample_count[angle_index] += 1

        for j in range(181):
            if sample_count[j] != 0:
                average_distance[j] = cumulative_distance[j] / sample_count[j]

        for k in range(181):
            if average_distance[k] != 0:
                risk_values[k] = 135.72 * math.exp(-0.6109 * average_distance[k])

        for k in range(181):
            if risk_values[k] >= self.max_risk_threshold:
                set_risk_zone(risk_map, k, 23)

        safe_angles = np.where(risk_map == 0)[0].tolist()
        # Waypoint로부터 계산된 목표방위각 사용
        heading_diff = float(self.target_imu_angle - self.imu_heading)
        desired_heading = self.target_imu_angle + heading_diff - (180.0 if self.target_imu_angle >= 180.0 else 0)

        heading = self.target_imu_angle  # 기본값은 목표방위각
        # 위험영역이 아닌 곳으로 heading 보정
        if safe_angles:
            heading = float(min(safe_angles, key=lambda x: abs(x - desired_heading)))
            heading = max(45.0, min(135.0, heading))

        self.key_target_degree = heading
        self.get_logger().info(f"key: {self.key_target_degree:5.1f}, IMU: {self.imu_heading:5.1f}, Target: {self.target_imu_angle:5.1f}")

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
