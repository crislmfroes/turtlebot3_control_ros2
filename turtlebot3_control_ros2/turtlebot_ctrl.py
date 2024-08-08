#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import networkx as nx
from transforms3d.euler import quat2euler, euler2quat
import math
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import tf2_ros
from rclpy.clock import ClockType
import random
from copy import deepcopy

from example_interfaces.msg import String

class TurtlebotCtrl(Node):
	def __init__(self):
		super().__init__("TurtlebotCtrl")

		self.linear = 0.0
		self.angular = 0.0
		self.state = 'path_planning'

		self.tf_buffer = Buffer()
		self.tfl = TransformListener(buffer=self.tf_buffer, node=self)
		self.goal_hash = None
		self.path = []
		self.heuristic_value = math.inf

		self.laser = LaserScan()
		self.odom = Odometry()

		self.map = np.array([	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
								[0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
						])
		self.publish_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
		self.subscriber_odom = self.create_subscription(Odometry, "/odom", self.callback_odom, 10)
		self.subscriber_laser = self.create_subscription(LaserScan, "/scan", self.callback_laser, rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=10))
		self.timer = self.create_timer(1/50.0, self.cmd_vel_pub)

	def cmd_vel_pub(self):

		map_resolution = 4

		index_x = -int(self.odom.pose.pose.position.x*map_resolution)
		index_y = -int(self.odom.pose.pose.position.y*map_resolution)

		index_x += int(self.map.shape[0]/2)
		index_y += int(self.map.shape[0]/2)

		if (index_x < 1): index_x = 1
		if (index_x > self.map.shape[0]-1): index_x = self.map.shape[0]-1
		if (index_y < 1): index_y = 1
		if (index_y > self.map.shape[0]-1): index_y = self.map.shape[0]-1

		if (self.map[index_x][index_y] == 1):
			self.map[index_x][index_y] = 2

			self.get_logger().info("Another part reached ... percentage total reached...." + str(100*float(np.count_nonzero(self.map == 2))/(np.count_nonzero(self.map == 1) + np.count_nonzero(self.map == 2))) )
			self.get_logger().info("Discrete Map")
			self.get_logger().info("\n"+str(self.map))
			self.get_logger().info(f"Goal: {self.goal_hash}")
			self.get_logger().info(f"Current position in grid: {index_x};{index_y}")
			self.get_logger().info(f"Current state: {self.state}")


        # Desenvlva seu codigo aqui
		
		graph = nx.Graph()

		if len(self.laser.ranges) == 0:
			return
		laser = deepcopy(self.laser)
		
		for i in range(self.map.shape[0]):
			for j in range(self.map.shape[1]):
					if self.map[i,j] != 0:
						weight = 1
					else:
						weight = 1000000
					try:
						if self.map[i,j+1] != 0:
							graph.add_edge(self.hash_cell(i, j), self.hash_cell(i, j+1), weight=weight)#, weight=np.sum(self.map[i-5,j-5].flatten() < 1))
						if self.map[i,j-1] != 0:
							graph.add_edge(self.hash_cell(i, j), self.hash_cell(i, j-1), weight=weight)#, weight=np.sum(self.map[i-5,j-5].flatten() < 1))
						if self.map[i+1,j] != 0:
							graph.add_edge(self.hash_cell(i, j), self.hash_cell(i+1, j), weight=weight)#, weight=np.sum(self.map[i-5,j-5].flatten() < 1))
						if self.map[i-1,j] != 0:
							graph.add_edge(self.hash_cell(i, j), self.hash_cell(i-1, j), weight=weight)#, weight=np.sum(self.map[i-5,j-5].flatten() < 1))
					except BaseException as e:
						pass

		all_cells = [self.unhash_cell(n) for n in graph.nodes]
		all_cells = sorted(all_cells, key=lambda e: math.dist(e, [index_x, index_y]))
		start_hash = self.hash_cell(*all_cells[0])
		if self.state == "path_planning":
			if len(self.path) == 0:
				self.goal_hash = None
			if (start_hash == self.goal_hash) or (self.goal_hash == None):
				self.goal_hash = None
				candidate_goals = []
				for i in range(self.map.shape[0]):
					for j in range(self.map.shape[1]):
						if self.map[i,j] == 1:
							goal_hash = self.hash_cell(i, j)
							if goal_hash in graph:
								candidate_goals.append(goal_hash)
							break
				self.goal_hash = random.choice(candidate_goals)
				try:
					path = nx.astar_path(graph, start_hash, self.goal_hash)
				except nx.NetworkXNoPath as e:
					raise e
					self.goal_hash = None
					return
				path = [self.unhash_cell(h) for h in path]
				path = [self.cell_to_point(*c) for c in path]
				path.pop(0)
				self.path = path
				self.start_time = rclpy.time.Time()
				self.state = 'move_to_goal'
		
		rot_threshsold = 0.05
		kp_angular = 1.0
		kp_linear = 1.0

		roll, pitch, yaw = quat2euler([
			self.odom.pose.pose.orientation.w,
			self.odom.pose.pose.orientation.x,
			self.odom.pose.pose.orientation.y,
			self.odom.pose.pose.orientation.z,
		])

		max_vel_linear = 0.1
		max_vel_angular = 0.5

		laser = deepcopy(self.laser)
		continuity_threshold = 0.25
		distance_threshold = 0.5
		obstacles = {}
		obstacle_index = 0
		for i in range(len(laser.ranges)):
			obstacle_index = int(obstacle_index)
			first = laser.ranges[i]
			try:
				last = laser.ranges[i+1]
			except BaseException as e:
				continue
			if abs(first - last) > continuity_threshold and last < distance_threshold:
				if obstacle_index not in obstacles:
					obstacles[obstacle_index] = []
				obstacles[obstacle_index].append(last)
			if abs(first - last) > continuity_threshold and first < distance_threshold:
				obstacle_index += 1
		base_point = PointStamped()

		try:
			transform = self.tf_buffer.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
		except BaseException as e:
			return
		base_point = do_transform_point(base_point, transform)
		a = (self.path[0][1] - base_point.point.y)/(self.path[0][0] - base_point.point.x)
		b = base_point.point.y - a*base_point.point.x
		obstacles_in_path = []
		for obstacle in obstacles:
			for obstacle_range_index in obstacles[obstacle]:
				obstacle_range = laser.ranges[int(obstacle_range_index)]
				theta = laser.angle_min + laser.angle_increment*obstacle_range_index
				laser_point = PointStamped()
				try:
					transform = self.tf_buffer.lookup_transform('odom', 'base_scan', rclpy.time.Time())
				except:
					return
				laser_point = do_transform_point(laser_point, transform)
				if np.allclose(a*laser_point.point.x + b, laser_point.point.y, rtol=0.1):
					if math.dist(self.path[0], [base_point.point.x, base_point.point.y]) > math.dist([laser_point.point.x, laser_point.point.y], [base_point.point.x, base_point.point.y]):
						obstacles_in_path.append(obstacle)
						break
		if self.state == 'move_to_goal':
			if len(obstacles_in_path) == 0:
				goal = self.path[0]
			else:
				end_points = []
				for obstacle in obstacles:
					for point_index in [0, -1]:
						obstacle_range_index = obstacles[obstacle][point_index]
						obstacle_range = laser.ranges[int(obstacle_range_index)]
						theta = laser.angle_min + laser.angle_increment*obstacle_range_index
						laser_point = PointStamped()
						try:
							transform = self.tf_buffer.lookup_transform('odom', 'base_scan', rclpy.time.Time())
						except:
							return
						laser_point = do_transform_point(laser_point, transform)
						end_points.append([laser_point.point.x, laser_point.point.y])
				end_points = sorted(end_points, key=lambda oi: math.dist([base_point.point.x, base_point.point.y], oi) + math.dist(oi, self.path[0]))
				goal = end_points[0]
				new_heuristic_value = math.dist([base_point.point.x, base_point.point.y], goal) + math.dist(goal, self.path[0])
				if new_heuristic_value > self.heuristic_value:
					self.obstacle_points = []
					self.state = 'move_around_obstacle'
				self.heuristic_value = new_heuristic_value
			point = PointStamped()

			try:
				transform = self.tf_buffer.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
			except BaseException as e:
				return
			point = do_transform_point(point, transform)
			self.a = (goal[1] - point.point.y)/(goal[0] - point.point.x)
			self.b = point.point.y - self.a*point.point.x
			if len(self.path) > 0:
				n_points = len(self.laser.ranges)
				middle_point = n_points//2
				left_ranges = self.laser.ranges[:middle_point]
				right_ranges = self.laser.ranges[middle_point:]
				front_ranges = left_ranges[:middle_point//2] + right_ranges[middle_point//2:]
				back_ranges = left_ranges[middle_point//2:] + right_ranges[:middle_point//2]
				dy = goal[1]-self.odom.pose.pose.position.y
				dx = goal[0]-self.odom.pose.pose.position.x
				setpoint_angle = math.atan2(dy, dx)
				if setpoint_angle > math.radians(180.0):
					setpoint_angle = -(math.radians(360.0)-setpoint_angle)
				
				point = PointStamped()
				point.point.x = goal[0]
				point.point.y = goal[1]
				point.point.z = 0.0

				try:
					transform = self.tf_buffer.lookup_transform('base_footprint', 'odom', rclpy.time.Time())
				except BaseException as e:
					return
				point = do_transform_point(point, transform)
				if point.point.y > 0:
					self.angular = max_vel_angular
				if point.point.y < 0:
					self.angular = -max_vel_angular
				if point.point.x > 0:
					self.linear = max_vel_linear
				if self.hash_cell(*self.point_to_cell(*goal)) == self.hash_cell(index_x, index_y):
					self.linear = 0.0
					self.angular = 0.0
					self.path.pop(0)
			if len(self.path) == 0:
				self.state = 'path_planning'
		if self.state == 'move_around_obstacle':
			front_ranges = laser.ranges[:45] + laser.ranges[-45:]
			left_ranges = laser.ranges[45:135]
			right_ranges = laser.ranges[-135:-45]
			if min(left_ranges) < distance_threshold and min(front_ranges) > distance_threshold:
				self.linear = max_vel_linear
				self.angular = 0.0
			elif min(left_ranges) < distance_threshold and min(front_ranges) < distance_threshold:
				self.linear = max_vel_linear
				self.angular = -max_vel_angular
			elif min(left_ranges) > distance_threshold and min(front_ranges) < distance_threshold:
				self.linear = 0.0
				self.angular = max_vel_angular
			else:
				self.linear = max_vel_linear
				self.angular = -max_vel_angular
			self.obstacle_points.append([base_point.point.x, base_point.point.y])
			end_points = []
			for obstacle in obstacles:
				for point_index in [0, -1]:
					obstacle_range_index = obstacles[obstacle][point_index]
					obstacle_range = laser.ranges[int(obstacle_range_index)]
					theta = laser.angle_min + laser.angle_increment*obstacle_range_index
					laser_point = PointStamped()
					try:
						transform = self.tf_buffer.lookup_transform('odom', 'base_scan', rclpy.time.Time())
					except:
						return
					laser_point = do_transform_point(laser_point, transform)
					end_points.append([laser_point.point.x, laser_point.point.y])
			d_followed = min([math.dist(p, self.path[0]) for p in self.obstacle_points])
			if len(end_points) == 0:
				d_reach = math.dist(self.path[0], [base_point.point.x, base_point.point.y])
			else:
				d_reach = min([math.dist(p, self.path[0]) for p in end_points])
			if d_reach <= d_followed:
				self.obstacle_points = []
				self.state = 'move_to_goal'
				self.heuristic_value = math.inf

		msg = Twist()
		msg.linear.x = self.linear
		msg.angular.z = self.angular
		self.publish_cmd_vel.publish(msg)
		

	def hash_cell(self, i, j):
		return f"{i};{j}"

	def unhash_cell(self, hash):
		return [int(v) for v in hash.split(";")]

	def cell_to_point(self, i, j):
		return -(i*(4.0/20.0) - 2.0), -(j*(4.0/20.0) - 2.0)

	def point_to_cell(self, x, y):
		map_resolution = 4

		index_x = -int(x*map_resolution)
		index_y = -int(y*map_resolution)

		index_x += int(self.map.shape[0]/2)
		index_y += int(self.map.shape[0]/2)

		if (index_x < 1): index_x = 1
		if (index_x > self.map.shape[0]-1): index_x = self.map.shape[0]-1
		if (index_y < 1): index_y = 1
		if (index_y > self.map.shape[0]-1): index_y = self.map.shape[0]-1

		return index_x, index_y

	def callback_laser(self, msg):
		self.laser = msg

	def callback_odom(self, msg):
		self.odom = msg

def main(args=None):
	rclpy.init(args=args)
	node = TurtlebotCtrl()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == "__main__":
	main()
