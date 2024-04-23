import sys
import rospy
import moveit_commander
import copy
import numpy as np

from geometry_msgs.msg import Pose
from moveit_msgs.msg import Constraints, OrientationConstraint

class Arm:

    def __init__(self):

        moveit_commander.roscpp_initialize(sys.argv)
        self.arm_group = moveit_commander.MoveGroupCommander('arm')
        self.gripper_group = moveit_commander.MoveGroupCommander('gripper')
        self.arm_group.set_workspace([0., -0.35, -0.02, 0.7, 0.35, 0.35])
        self.camera_pose = [0., 50., 52., -93.26, -138.26, 90.]
        self.home_pose  = [0., -0.28, 1.309, 0., -1.047, 0.]
        self.ready_pose = [0., 0., 90., -90., -70., 90.]
        self.ready_pose_alt = [19.7, -29.5, 72.13, -88.64, -80.19, 103.47]
        self.new_ready_pose = [-98.44, 25.77, -127.56, 26.12, -74.25, -119.43] #[261.56, 25.77, 232.44, 26.12, 285.75, 240.57]
        self.camera_pose_alt = [90.0, -90.0, 0., 0., 0., 0.]

    def get_pose(self):

        pose = self.arm_group.get_current_pose().pose
        return pose

    def to_home_pose(self):

        self.arm_group.set_joint_value_target(self.home_pose)
        self.arm_group.go(wait=True)
        return

    def to_ready_pose(self):

        ready_pose = (np.array(self.ready_pose) * np.pi / 180.).tolist()
        self.arm_group.set_joint_value_target(ready_pose)
        self.arm_group.go(wait=True)
        return

    def to_ready_pose_alt(self):

        ready_pose = (np.array(self.ready_pose_alt) * np.pi / 180.).tolist()
        self.arm_group.set_joint_value_target(ready_pose)
        self.arm_group.go(wait=True)
        return

    def to_new_ready_pose(self):

        ready_pose = (np.array(self.new_ready_pose) * np.pi / 180.).tolist()
        self.arm_group.set_joint_value_target(ready_pose)
        self.arm_group.go(wait=True)
        return

    def to_camera_pose(self):

        camera_pose = (np.array(self.camera_pose) * np.pi / 180.).tolist()
        self.arm_group.set_joint_value_target(camera_pose) 
        self.arm_group.go(wait=True)
        return

    def to_camera_pose_alt(self):

        camera_pose = (np.array(self.camera_pose_alt) * np.pi / 180.).tolist()
        self.arm_group.set_joint_value_target(camera_pose) 
        self.arm_group.go(wait=True)
        return

    '''
    Quaternion
    EEF perpendicular & finger direction is | : [0, 1, 0, 0]
    EEF perpendicular & finger direction is -- : [sqrt(0.5), sqrt(0.5), 0, 0]
    '''
    def eef_orientation(self, qx, qy, qz, qw):

        wpose = self.get_pose()
        # orientation
        target = Pose()
        target.position = wpose.position
        target.orientation.x = qx
        target.orientation.y = qy
        target.orientation.z = qz
        target.orientation.w = qw
        plan = self.arm_group.plan(target)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.1)
        return

    def cartesian_target_default(self, x, y, z, action='pick'):

        rospy.loginfo('Cartesian target position: X:{} Y:{} Z:{}'.format(x, y, z))
        pose = self.arm_group.get_current_pose().pose
        # forward xy
        target_xy = Pose()
        target_xy.orientation = pose.orientation
        target_xy.position.x = x
        target_xy.position.y = y
        target_xy.position.z = pose.position.z
        plan = self.arm_group.plan(target_xy)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        # forward z down
        target_z = Pose()
        target_z.orientation = pose.orientation
        target_z.position = copy.deepcopy(target_xy.position)
        target_z.position.z = z
        plan = self.arm_group.plan(target_z)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        # pick or place
        if action == 'pick':
            self.finger_close()
        elif action == 'place':
            self.finger_open()
        # backward z up
        plan = self.arm_group.plan(target_xy)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        # backward xy move
        plan = self.arm_group.plan(pose)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        return

    def cartesian_target_quadrant2(self, x, y, z, action='pick'):

        rospy.loginfo('Cartesian target position: X:{} Y:{} Z:{}'.format(x, y, z))
        self.to_ready_pose_alt()
        rospy.sleep(1.0)
        pose = self.arm_group.get_current_pose().pose
        # forward xy
        target_xy = Pose()
        target_xy.orientation = pose.orientation
        target_xy.position.x = x
        target_xy.position.y = y
        target_xy.position.z = pose.position.z
        plan = self.arm_group.plan(target_xy)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        # forward z down
        target_z = Pose()
        target_z.orientation = pose.orientation
        target_z.position = copy.deepcopy(target_xy.position)
        target_z.position.z = z
        plan = self.arm_group.plan(target_z)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        # pick or place
        if action == 'pick':
            self.finger_close()
        elif action == 'place':
            self.finger_open()
        # backward z up
        plan = self.arm_group.plan(target_xy)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        # backward xy move
        '''
        plan = self.arm_group.plan(pose)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(1.0)
        '''
        self.to_ready_pose()
        return

    def cartesian_target_quadrant4(self, x, y, z, action='pick'):

        rospy.loginfo('Cartesian target position: X:{} Y:{} Z:{}'.format(x, y, z))
        pose = self.arm_group.get_current_pose().pose
        # Forward XY
        target_xy = Pose()
        target_xy.orientation = pose.orientation
        target_xy.position.x = x
        target_xy.position.y = y
        target_xy.position.z = 0.15
        plan = self.arm_group.plan(target_xy)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        # Forward Z down
        target_z = Pose()
        target_z.orientation = pose.orientation
        target_z.position = copy.deepcopy(target_xy.position)
        target_z.position.z = z
        plan = self.arm_group.plan(target_z)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        if action == 'pick':
            self.finger_close()
        elif action == 'place':
            self.finger_open()
        # Backward Z up
        plan = self.arm_group.plan(target_xy)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        # Backward XY move
        plan = self.arm_group.plan(pose)
        self.arm_group.execute(plan, wait=True)
        rospy.sleep(0.3)
        return

    def finger_open(self):

        self.gripper_group.set_joint_value_target([-0.80, 0., 0.80, 0.])
        self.gripper_group.go(wait=True)
        return

    def finger_close(self):

        self.gripper_group.set_joint_value_target([-0.03, 0., 0.03, 0.])
        self.gripper_group.go(wait=True)
        return


if __name__ == '__main__':

    from util import random_orientation_sample
    rospy.init_node('arm_test')
    arm = Arm()  
    '''
    arm.to_ready_pose()
    print(arm.arm_group.get_current_pose())
    rospy.sleep(1.0)
    arm.to_camera_pose()
    rospy.sleep(1.0)
    '''
    #arm.to_home_pose()
    arm.to_camera_pose()
    arm.finger_open()
    '''
    K = 4
    pose = arm.get_pose()
    current_ori = [pose.orientation.x,
                   pose.orientation.y,
                   pose.orientation.z,
                   pose.orientation.w]
    sampled_ori_array = random_orientation_sample(current_ori, K)
    for ori in sampled_ori_array:
        arm.eef_orientation(*ori)
        print(arm.get_pose())
    arm.to_ready_pose()
    '''





