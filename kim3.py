#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter

import math
import time
import numpy as np
from numpy import sin, cos, arccos, pi

# 메시지
from sensor_msgs.msg import LaserScan, NavSatFix
from mechaship_interfaces.msg import RgbwLedColor

class Hopping(Node):
    def __init__(self):
        super().__init__(
            "hopping_node",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.get_logger().info("----- start Hoping node (Rectangular Track ver.) -----")

        # 초기 헤딩 (yaw) 설정
        self.init_yaw_heading = -1
        self.now_heading = 0
       
        # 기본 키(조향각), 스로틀
        self.final_key_angle = 90
        self.throttle_value = 50

        # 시작 지점 GPS
        self.init_gps = [35.23196029663086, 129.08291625976562]
        self.delta_latitude = 0
        self.delta_longitude = 0

        # 사각형 경기장 웨이포인트 (예시)
        corner1 = [35.178849, 128.554747]
        corner2 = [35.232000, 129.083000]
        corner3 = [35.231900, 129.083000]
        corner4 = [35.231900, 129.082900]
        self.goal_gps_list = [
            corner1,
            corner2,
            corner3,
            corner4,
            corner1  # 한 바퀴 돌고 복귀
        ]

        # 현재 위치, 목표 지점
        self.now_gps = [35.178849, 128.554747]
        self.goal_gps = self.goal_gps_list[0]
        self.distance_to_goal = 0

        # LED 점등 관련 시간/플래그
        self.timestamp = -1

        # 장애물 감지 여부
        self.obstacle_detected = False

        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 구독: Heading(NavSatFix 형태로 가정), GPS, LiDAR
        self.now_heading_subscription = self.create_subscription(
            NavSatFix, "heading", self.heading_listener_callback, qos_profile
        )
        self.gps_subscription = self.create_subscription(
            NavSatFix, "gps/data", self.gps_listener_callback, qos_profile
        )
        self.scan_subscription = self.create_subscription(
            LaserScan, "scan", self.lidar_listener_callback, qos_profile
        )

        # LED 퍼블리셔
        self.led_pub = self.create_publisher(
            RgbwLedColor, "/actuator/rgbwled/color", 10
        )

        # 초기 GPS 보정
        self.gps_calibration()

    def gps_calibration(self):
        """
        실행 시 콘솔에서 "input neo_6m's now gps data : "에 현재 장비 GPS를 입력하면,
        init_gps와의 차이를 구해 goal_gps_list를 보정.
        """
        neo_6m_data = input("input neo_6m's now gps data : ").split(',')
        neo_latitude, neo_longitude = map(float, neo_6m_data)
        self.delta_latitude = neo_latitude - self.init_gps[0]
        self.delta_longitude = neo_longitude - self.init_gps[1]
       
        for idx in range(len(self.goal_gps_list)):
            self.goal_gps_list[idx][0] += self.delta_latitude
            self.goal_gps_list[idx][1] += self.delta_longitude

        self.goal_gps = self.goal_gps_list[0]

    def heading_listener_callback(self, data: NavSatFix):
        """
        Heading 콜백 (NavSatFix 기반):
        yaw값을 data.longitude로 임시 사용.
        """
        self.now_heading = data.longitude
        if self.init_yaw_heading == -1:
            self.init_yaw_heading = self.now_heading
            self.goal_gps = self.goal_gps_list[0]
            self.get_logger().info(f"yaw 초기화: {self.init_yaw_heading}")

        # 초기값 빼서 상대 0도로 맞춤
        self.now_heading -= self.init_yaw_heading

    def gps_listener_callback(self, data: NavSatFix):
        """
        GPS 콜백:
        현재 위치(now_gps) 갱신 후 거리 계산,
        스로틀/조향각 제어 로직.
        """
        self.now_gps = [data.latitude, data.longitude]
        self.cal_distance()
        self.drive_to_goal_throttle()
        self.drive_to_goal_key()

    def lidar_listener_callback(self, msg: LaserScan):
        """
        LiDAR 콜백: 최소 거리 판단 -> 1.5m 미만이면 장애물 감지.
        """
        if not msg.ranges:
            return

        min_dist = min(msg.ranges)
        if min_dist < 1.5:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def cal_distance(self):
        """
        now_gps와 goal_gps 간 거리(미터)를 계산해 distance_to_goal에 저장
        """
        lat1, lon1 = self.now_gps
        lat2, lon2 = self.goal_gps
        theta = lon1 - lon2
        dist = 60 * 1.1515 * self.rad2deg(
            arccos(
                (sin(self.deg2rad(lat1)) * sin(self.deg2rad(lat2))) +
                (cos(self.deg2rad(lat1)) * cos(self.deg2rad(lat2)) * cos(self.deg2rad(theta)))
            )
        )
        self.distance_to_goal = dist * 1.609344 * 1000
        self.get_logger().info(f"distance to goal = {self.distance_to_goal:.2f} m")

    def drive_to_goal_throttle(self):
        """
        1) LiDAR 장애물 -> 노란색 LED, 정지(또는 저속)
        2) 3번 웨이포인트 도착 시: 파랑 -> 20초 -> 빨강 -> 5초 -> 노랑
        3) 거리 범위별로 전진/정지
        """
        # 장애물 감지 시 노란 LED
        if self.obstacle_detected:
            self.activate_led(r=254, g=254, b=0, w=0)
            self.get_logger().info("장애물 감지 - 노란색 LED, 정지 혹은 저속")
            # 스로틀 로직(멈추거나 저속)
            return

        # 목표까지 거리 로직
        if self.distance_to_goal < 1.5:
            idx = self.goal_gps_list.index(self.goal_gps) + 1
            self.get_logger().info(f"{idx}번째 웨이포인트 근접!")
            # 3번 웨이포인트일 때
            if idx == 3:
                self.activate_led(r=0, g=0, b=254, w=0)  # 파랑
                time.sleep(20)
                self.activate_led(r=254, g=0, b=0, w=0)  # 빨강
                time.sleep(5)
                self.activate_led(r=254, g=254, b=0, w=0)  # 노랑

            # 다음 웨이포인트로 이동
            if len(self.goal_gps_list) > 0:
                self.goal_gps_list.pop(0)
                if len(self.goal_gps_list) == 0:
                    self.get_logger().info("모든 경로 완료!")
                else:
                    self.goal_gps = self.goal_gps_list[0]
            return

        elif 1.5 <= self.distance_to_goal < 4.0:
            # 중간 거리 -> 빨간색
            self.activate_led(r=254, g=0, b=0, w=0)
        else:
            self.get_logger().info(f"목표 {self.goal_gps}로 전진")

    def drive_to_goal_key(self):
        """
        보트의 현재 헤딩(now_heading)과
        '현재 위치->목표 위치' 방위각의 차를 보고
        +10도 이상 -> 우회전, -10도 이하 -> 좌회전, 나머지 -> 전진
        """
        lat1, lon1 = self.now_gps
        lat2, lon2 = self.goal_gps
        target_angle = self.azimuth(lat1, lon1, lat2, lon2)
        diff_angle = target_angle - self.now_heading

        if diff_angle > 10:
            self.get_logger().info("우측 선회")
        elif diff_angle < -10:
            self.get_logger().info("좌측 선회")
        else:
            self.get_logger().info("직진")

    def activate_led(self, r=0, g=0, b=0, w=0):
        """
        /actuator/rgbwled/color 토픽에 RgbwLedColor 메시지를 발행
        r, g, b, w는 모두 0~255 범위
        """
        led_msg = RgbwLedColor()
        led_msg.red = r
        led_msg.green = g
        led_msg.blue = b
        led_msg.white = w
        self.led_pub.publish(led_msg)

    def rad2deg(self, radians):
        return radians * 180.0 / pi

    def deg2rad(self, degrees):
        return degrees * pi / 180.0

    def azimuth(self, lat1, lng1, lat2, lng2):
        """
        두 점(lat1, lng1), (lat2, lng2) 사이 방위각(deg)
        """
        Lat1 = math.radians(lat1)
        Lat2 = math.radians(lat2)
        Lng1 = math.radians(lng1)
        Lng2 = math.radians(lng2)
        y = math.sin(Lng2 - Lng1) * math.cos(Lat2)
        x = (math.cos(Lat1)*math.sin(Lat2)
            - math.sin(Lat1)*math.cos(Lat2)*math.cos(Lng2 - Lng1))
        z = math.atan2(y, x)
        a = math.degrees(z)
        if a < 0:
            a += 360.0
        return a

def main(args=None):
    rclpy.init(args=args)
    node = Hopping()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt -> 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
