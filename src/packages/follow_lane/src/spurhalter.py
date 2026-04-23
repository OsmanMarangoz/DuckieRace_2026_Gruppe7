#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import Int32, Float64
from duckietown_msgs.msg import Twist2DStamped

try:
    from switch_control_node import ControlType
except Exception:
    class ControlType:
        Lane = 1


class spurhalter:
    """PID-basierter Line Follower:
    - Ziel: error = 0 (Spur in der Bildmitte)
    - Empfängt error von detect_lane_node: ~[-1, 1]
    - Berechnet Lenkbefehle mit PID-Regler
    - Fährt wenn Spur sichtbar, sucht wenn verloren
    """

    def __init__(self, node_name='spurhalter'):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ.get('VEHICLE_NAME', '')

        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cb_lane, queue_size=1)
        self.sub_control = rospy.Subscriber(f"/{self._vehicle_name}/switch/control", Int32, self.cb_control, queue_size=1)

        # PID-Parameter
        self.kp = rospy.get_param('~kp', 1.0)      # Proportional
        self.ki = rospy.get_param('~ki', 0.1)      # Integral
        self.kd = rospy.get_param('~kd', 0.5)      # Derivative

        # Geschwindigkeit
        self.max_v = rospy.get_param('~max_v', 0.4)
        self.min_v = rospy.get_param('~min_v', 0.05)
        self.max_omega = rospy.get_param('~max_omega', 2.0)

        # Suchverhalten bei verlorener Spur
        self.search_omega = rospy.get_param('~search_omega', 0.5)
        self.max_error_threshold = rospy.get_param('~max_error_threshold', 1.5)

        # PID-State
        self.error_integral = 0.0
        self.last_error = 0.0
        self.enabled = True
        self.rate = rospy.Rate(10)
        rospy.on_shutdown(self.shutdown)

    def cb_control(self, msg: Int32):
        try:
            self.enabled = (msg.data == ControlType.Lane.value)
        except Exception:
            self.enabled = (msg.data == 1)

    def cb_lane(self, msg: Float64):
        if not self.enabled:
            return

        import math
        e = float(msg.data)
        if not math.isfinite(e):
            return

        # Prüfe: ist Spur sichtbar? (error im Normalbereich)
        if abs(e) > self.max_error_threshold:
            # Spur verloren: suche durch Drehung
            v = 0.0
            omega = self.search_omega if e >= 0 else -self.search_omega
        else:
            # Spur sichtbar: fahre mit PID-Regler
            dt = 0.1  # 10 Hz

            # P: proportional zum Fehler
            p = self.kp * e

            # I: Integral des Fehlers über die Zeit
            self.error_integral += e * dt
            i = self.ki * self.error_integral

            # D: Ableitung des Fehlers
            d = self.kd * (e - self.last_error) / dt

            # Lenkbefehl
            omega = p + i + d
            omega = max(-self.max_omega, min(self.max_omega, omega))

            # Geschwindigkeit reduzieren bei großem Fehler
            v = self.max_v * (1.0 - abs(e))
            v = max(self.min_v, v)

            self.last_error = e

        # Sende Befehl
        twist = Twist2DStamped()
        twist.header.stamp = rospy.Time.now()
        twist.v = float(v)
        twist.omega = float(omega)
        self.pub_cmd.publish(twist)

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def shutdown(self):
        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd.publish(twist)


if __name__ == '__main__':
    node = spurhalter()
    node.run()
