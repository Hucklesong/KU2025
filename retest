import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan, Imu, NavSatFix
from math import radians, degrees, atan2, cos, sin
from tf_transformations import euler_from_quaternion
import numpy as np
from sklearn.cluster import DBSCAN

from mechaship_interfaces.msg import RgbwLedColor

def haversine(lat1, lon1, lat2, lon2):
    # 위도,경도(도) → 라디안
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000 * c  # 지구 반지름(m)

def bearing(lat1, lon1, lat2, lon2):
    # 두 지점의 방위각(진행 각도)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    angle = np.arctan2(x, y)
    return (np.degrees(angle) + 360) % 360

class NavigationNode(Node):
    def __init__(self):
        super().__init__("navigation_node", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.front_lidar = []
        self.yaw_degree = 0.0
        self.current_lat = None
        self.current_lon = None
        self.current_waypoint_index = 0
        # 예시 경유점 (위도, 경도) - 실제 경유점으로 교체
        self.waypoints = [
            (35.123456, 128.123456),
            (35.124000, 128.124000)
        ]

        # Subscribers
        self.lidar_subscription = self.create_subscription(LaserScan, "/scan", self.lidar_callback, qos_profile_sensor_data)
        self.imu_subscription = self.create_subscription(Imu, "/imu", self.imu_callback, qos_profile_sensor_data)
        self.gps_subscription = self.create_subscription(NavSatFix, "/gps/fix", self.gps_callback, qos_profile_sensor_data)

        # Publishers
        self.thruster_publisher = self.create_publisher(Float32, "/actuator/thruster/percentage", 10)
        self.servo_publisher = self.create_publisher(Float32, "/actuator/key/degree", 10)
        self.rgbw_led_publisher = self.create_publisher(RgbwLedColor, "/actuator/rgbwled/color", 10)

        self.create_timer(0.5, self.timer_callback)

    def lidar_callback(self, data):
        total_ranges = len(data.ranges)
        fov_degrees = 160
        start_index = (total_ranges // 2) - (total_ranges * fov_degrees // 360 // 2)
        end_index = (total_ranges // 2) + (total_ranges * fov_degrees // 360 // 2)
        self.front_lidar = data.ranges[start_index:end_index]

    def imu_callback(self, msg):
        q = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        _, _, yaw_rad = euler_from_quaternion(q)
        self.yaw_degree = (degrees(yaw_rad) + 360) % 360

    def gps_callback(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    def timer_callback(self):
        # 경유점 도달 여부 확인
        if self.current_lat is None or self.current_lon is None:
            return

        wp_lat, wp_lon = self.waypoints[self.current_waypoint_index]
        dist = haversine(self.current_lat, self.current_lon, wp_lat, wp_lon)
        if dist < 3.0:  # 3m 이내면 다음 경유점
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                self.get_logger().info(f"Waypoint {self.current_waypoint_index} 도달, 다음 경유점으로 이동")
            else:
                self.get_logger().info("최종 목적지 도달! 정지합니다.")
                self.set_thruster_and_servo(0.0, 90.0)
                return

        filtered_lidar = self.filter_lidar_data(self.front_lidar)
        clusters = self.cluster_lidar_data(filtered_lidar)

        if clusters:  # 장애물 있음
            angle = self.calculate_avoidance_angle(clusters)
            self.get_logger().info(f"장애물 회피: 각도 {angle}")
            self.set_thruster_and_servo(25.0, 90.0 + angle)  # 각도 조정값 심플하게 처리
        else:  # 장애물 없음: 경유점 방향 추종
            target_bearing = bearing(self.current_lat, self.current_lon, wp_lat, wp_lon)
            error = ((target_bearing - self.yaw_degree + 540) % 360) - 180  # -180~+180 차이
            self.get_logger().info(f"경유점 추종: 목표 방위각 {target_bearing}, 현재 {self.yaw_degree}, 오차 {error}")
            self.set_thruster_and_servo(25.0, 90.0 + error * 0.7)  # 보정값은 환경 맞춰 조절

    def filter_lidar_data(self, lidar_data):
        return [d for d in lidar_data if 0.1 < d < 10.0 and not np.isinf(d) and not np.isnan(d)]

    def cluster_lidar_data(self, lidar_data):
        if not lidar_data:
            return []

        total_ranges = len(lidar_data)
        angles = np.linspace(-80, 80, total_ranges)
        points = np.array([
            [d * np.cos(np.radians(a)), d * np.sin(np.radians(a))]
            for d, a in zip(lidar_data, angles) if d > 0
        ])

        db = DBSCAN(eps=0.5, min_samples=7).fit(points)
        labels = db.labels_

        clusters = {}
        for label in set(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            center = np.mean(cluster_points, axis=0)
            clusters[label] = center

        return clusters

    def calculate_avoidance_angle(self, clusters):
        # 가장 가까운 클러스터(장애물) 좌표 기준 왼쪽/오른쪽 선택 (간단 구현)
        closest = min(clusters.values(), key=lambda c: np.hypot(c[0], c[1]))
        angle = np.degrees(np.arctan2(closest[1], closest[0]))
        # 장애물 기준 반대 방향 (좌/우 중 더 넓은 쪽 실제 코드에선 safe_zone 사용 권장)
        return -45 if angle > 0 else 45

    def set_thruster_and_servo(self, thruster, key_degree):
        thruster_msg = Float32()
        thruster_msg.data = thruster
        self.thruster_publisher.publish(thruster_msg)

        servo_msg = Float32()
        servo_msg.data = key_degree
        self.servo_publisher.publish(servo_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 사용자 종료 요청 - Thruster/Servo 정지")
        # 안전 정지: Thruster 0, Servo 90(직진)
        thruster_msg = Float32()
        thruster_msg.data = 0.0
        node.thruster_publisher.publish(thruster_msg)
        servo_msg = Float32()
        servo_msg.data = 90.0
        node.servo_publisher.publish(servo_msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
