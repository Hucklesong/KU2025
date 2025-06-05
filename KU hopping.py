import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32
from geopy.distance import geodesic
import math

class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')
        # 경유점 리스트 [(lat, lon), ...]
        self.waypoints = [
            (37.12345, 127.12345),
            (37.12400, 127.12400),
            (37.12500, 127.12450)
        ]
        self.current_wp_idx = 0
        self.arrival_threshold = 5.0 # meters

        self.current_lat = None
        self.current_lon = None

        # 현재 위치 구독
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10
        )
        # (예시) Heading/속도 퍼블리시
        self.heading_pub = self.create_publisher(Float32, '/cmd_heading', 10)

    def gps_callback(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

        if self.current_wp_idx < len(self.waypoints):
            target_lat, target_lon = self.waypoints[self.current_wp_idx]
            dist = geodesic((self.current_lat, self.current_lon), (target_lat, target_lon)).meters
            bearing = self.calculate_bearing(self.current_lat, self.current_lon, target_lat, target_lon)
            self.get_logger().info(
                f'목표 경유점 {self.current_wp_idx+1}: 거리={dist:.1f}m, 방위각={bearing:.1f}deg'
            )

            # 선박 Heading 제어 명령
            heading_msg = Float32()
            heading_msg.data = bearing
            self.heading_pub.publish(heading_msg)

            # 도달했으면 다음 경유점으로 이동
            if dist < self.arrival_threshold:
                self.get_logger().info(f'경유점 {self.current_wp_idx+1} 도달!')
                self.current_wp_idx += 1
        else:
            self.get_logger().info('모든 경유점 도달 완료!')
            # 필요시 노드 종료: rclpy.shutdown()

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

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
