import rospy
import cv2
from camera import Camera
from arm import Arm
from util import vg_targets, vg_targets_grasp_only, PCA, floor_removal

class Agent:

    def __init__(self, test=False):

        self.arm = Arm()
        self.camera = Camera(test)

    def _cartesian_wrapper(self, x, y, z, action='pick'):

        if (y > -0.04) and (x < 0.44):
            # quadrant 2
            self.arm.cartesian_target_quadrant2(x, y, z, action)
        elif (y <= -0.04) and (x >= 0.5):
            # quadrant 4
            self.arm.cartesian_target_quadrant4(x, y, z, action)
        else:
            self.arm.cartesian_target_default(x, y, z, action)
        '''
        if (x >= 0.5):
            # quadrant 4
            self.arm.cartesian_target_quadrant4(x, y, z, action)
        else:
            self.arm.cartesian_target_default(x, y, z, action)
        '''
        return

    def pick_n_place(self, tl_x, tl_y, br_x, br_y, plx, ply, cloud, clean):

        pick_target, place_target = vg_targets(tl_x, tl_y, br_x, br_y, plx, ply, cloud, clean)
        # Open finger before pick-n-place
        self.arm.finger_open()
        rospy.sleep(0.1)
        # Go to pick target & grab it
        self._cartesian_wrapper(pick_target[0], pick_target[1], pick_target[2], 'pick')
        # Go to placement & place it
        self._cartesian_wrapper(place_target[0], place_target[1], place_target[2], 'place')
        return

    def grasp_only(self, tl_x, tl_y, br_x, br_y, cloud):

        pick_target = vg_targets_grasp_only(tl_x, tl_y, br_x, br_y, cloud)
        # Open finger before pick-n-place
        self.arm.finger_open()
        rospy.sleep(0.1)
        # Go to pick target & grab it
        self._cartesian_wrapper(pick_target[0], pick_target[1], pick_target[2], 'pick')
        # Release Object
        raw_input("Press any key to release object")
        self.arm.finger_open()
        return

    def grasp(self, tl_x, tl_y, br_x, br_y, cloud):

        pick_target = vg_targets_grasp_only(tl_x, tl_y, br_x, br_y, cloud)
        # Open finger before pick-n-place
        self.arm.finger_open()
        rospy.sleep(0.1)
        # Go to pick target & grab it
        self._cartesian_wrapper(pick_target[0], pick_target[1], pick_target[2], 'pick')
        # Release Object
        #raw_input("Press any key to release object")
        #self.arm.finger_open()
        return


if __name__ == '__main__':

    rospy.init_node('vg_demo_test', anonymous=False, disable_signals=True)
    agent = Agent(test=False)

    rospy.loginfo('Demo Start !!!')
    agent.arm.to_camera_pose()

    
    idx = 0
    while not rospy.is_shutdown():
        try:
            raw_input('Press any key')
            # Take visual info and send image to the server
            cv_img = agent.camera.get_rgb()
            # Send Captured Image
            cv2.imwrite('samples/{}.png'.format(idx), cv_img)
            cloud = agent.camera.get_cloud()
            idx += 1
        except KeyboardInterrupt:
            print('\nClient Ctrl-c')
            break
       
