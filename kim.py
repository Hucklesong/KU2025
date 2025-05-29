#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ROS2 관련 모듈 임포트
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter

# 기본 파이썬 모듈
import math
import time
import numpy as np
from numpy import sin, cos, arccos, pi

# ROS2 메시지 타입 임포트
from sensor_msgs.msg import LaserScan, NavSatFix
from mechaship_interfaces.msg import RgbwLedColor

class Hopping(Node):
    """사각형 경로(waypoint) 자율 주행 및 장애물 감지/회피 ROS2 노드"""

    def __init__(self):
        # ROS2 Node 초기화
        super().__init__(
            "hopping_node",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.get_logger().info("----- start Hoping node (Rectangular Track ver.) -----")

        # 방향, 스로틀, GPS 등 초기값 설정
        self.init_yaw_heading = -1      # 최초 기준 heading(방위각)
        self.now_heading = 0            # 현재 heading
        self.final_key_angle = 90       # (사용 안함)
        self.throttle_value = 50        # (사용 안함)

        # 기준 GPS 및 이동할 코너들(waypoint) 설정
        self.init_gps = [35.23196029663086, 129.08291625976562]
        self.delta_latitude = 0         # GPS 보정값
        self.delta_longitude = 0

        corner1 = [35.178849, 128.554747]
        corner2 = [35.232000, 129.083000]
        corner3 = [35.231900, 129.083000]
        corner4 = [35.231900, 129.082900]
        self.goal_gps_list = [
            corner1,
            corner2,
            corner3,
            corner4,
            corner1     # 출발점으로 돌아오는 루프
        ]

        self.now_gps = [35.178849, 128.554747]  # 현재 GPS 좌표
        self.goal_gps = self.goal_gps_list[0]   # 현재 목표점
        self.distance_to_goal = 0               # 목표점까지 거리

        self.timestamp = -1                     # (사용 안함)
        self.obstacle_detected = False          # 장애물 감지 여부

        # QoS 프로파일 설정 (센서 데이터용: 최신값만 수신)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ROS2 토픽 구독 설정
        self.now_heading_subscription = self.create_subscription(
            NavSatFix, "heading", self.heading_listener_callback, qos_profile
        )
        self.gps_subscription = self.create_subscription(
            NavSatFix, "gps/data", self.gps_listener_callback, qos_profile
        )
        self.scan_subscription = self.create_subscription(
            LaserScan, "scan", self.lidar_listener_callback, qos_profile
        )

        # LED 제어 퍼블리셔
        self.led_pub = self.create_publisher(
            RgbwLedColor, "/actuator/rgbwled/color", 10
        )

        # GPS 오프셋 보정 (초기화 시 사용자 입력 필요)
        self.gps_calibration()

    def gps_calibration(self):
        """
        실제 GPS 센서 좌표 입력받아 waypoint 전체를 상대적으로 보정
        """
        neo_6m_data = input("input neo_6m's now gps data : ").split(',')
        neo_latitude, neo_longitude = map(float, neo_6m_data)
        # 입력받은 GPS 기준으로 전체 waypoint 위치 보정
        self.delta_latitude = neo_latitude - self.init_gps[0]
        self.delta_longitude = neo_longitude - self.init_gps[1]
        for idx in range(len(self.goal_gps_list)):
            self.goal_gps_list[idx][0] += self.delta_latitude
            self.goal_gps_list[idx][1] += self.delta_longitude
        self.goal_gps = self.goal_gps_list[0]

    def heading_listener_callback(self, data: NavSatFix):
        """
        heading 토픽 콜백: 방위각(heading) 값 수신, 최초엔 초기 heading 저장
        """
        self.now_heading = data.longitude
        if self.init_yaw_heading == -1:
            self.init_yaw_heading = self.now_heading
            self.goal_gps = self.goal_gps_list[0]
            self.get_logger().info(f"yaw 초기화: {self.init_yaw_heading}")
        # 기준값에서 상대적으로 보정 (편차 계산)
        self.now_heading -= self.init_yaw_heading

    def gps_listener_callback(self, data: NavSatFix):
        """
        GPS 위치 토픽 콜백: 위치 업데이트 및 목표점까지 이동 판단
        """
        self.now_gps = [data.latitude, data.longitude]
        self.cal_distance()
        self.drive_to_goal_throttle()  # 거리, 장애물 상황에 따른 진행/정지
        self.drive_to_goal_key()       # 방향 조정(좌/우/직진)

    def lidar_listener_callback(self, msg: LaserScan):
        """
        Lidar 스캔 콜백: 장애물 감지(가장 가까운 거리 1.5m 미만시 장애물로 간주)
        """
        if not msg.ranges:
            return
        min_dist = min(msg.ranges)
        self.obstacle_detected = min_dist < 1.5

    def cal_distance(self):
        """
        현재 위치(now_gps)와 목표(goal_gps)까지 거리 계산 (단위: m)
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
        거리와 장애물 감지 상황에 따라 진행/정지/LED 색상 제어 및 waypoint 이동
        """
        if self.obstacle_detected:
            # 장애물 감지시: 노란 LED, 정지 또는 저속
            self.activate_led(r=254, g=254, b=0, w=0)
            self.get_logger().info("장애물 감지 - 노란색 LED, 정지 혹은 저속")
            return

        # 목표점 근접 시(1.5m 미만): 다음 waypoint로 이동
        if self.distance_to_goal < 1.5:
            idx = self.goal_gps_list.index(self.goal_gps) + 1
            self.get_logger().info(f"{idx}번째 웨이포인트 근접!")
            # 특정 waypoint에서는 LED 색상 시퀀스(예시)
            if idx == 3:
                self.activate_led(r=0, g=0, b=254, w=0)
                time.sleep(20)
                self.activate_led(r=254, g=0, b=0, w=0)
                time.sleep(5)
                self.activate_led(r=254, g=254, b=0, w=0)

            if len(self.goal_gps_list) > 0:
                self.goal_gps_list.pop(0)
            if len(self.goal_gps_list) == 0:
                self.get_logger().info("모든 경로 완료!")
            else:
                self.goal_gps = self.goal_gps_list[0]
            return

        elif 1.5 <= self.distance_to_goal < 4.0:
            # 가까우면 빨간색 LED
            self.activate_led(r=254, g=0, b=0, w=0)
        else:
            # 기본 전진(로그 출력만, 실제 주행 명령은 없음)
            self.get_logger().info(f"목표 {self.goal_gps}로 전진")

    def drive_to_goal_key(self):
        """
        현재 위치와 목표점 기준으로 방위각 차이 계산 → 좌/우/직진 판단
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
        RGBW LED 제어 메시지 발행
        """
        led_msg = RgbwLedColor()
        led_msg.red = r
        led_msg.green = g
        led_msg.blue = b
        led_msg.white = w
        self.led_pub.publish(led_msg)

    # 각도/라디안 변환 유틸 함수
    def rad2deg(self, radians):
        return radians * 180.0 / pi

    def deg2rad(self, degrees):
        return degrees * pi / 180.0

    def azimuth(self, lat1, lng1, lat2, lng2):
        """
        두 지점 사이의 방위각(azimuth) 계산 (도 단위)
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
    # ROS2 노드 및 Executor 생성
    rclpy.init(args=args)
    node = Hopping()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        # 노드 구동 (이벤트 루프)
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt -> 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
