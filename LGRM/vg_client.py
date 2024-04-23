import sys
import rospy
import socket
import cv2
import numpy as np
import base64

import time
from agent import Agent

HOST = '' # Host IP Address
PORT = 9998

def main():

    cli_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli_sock.connect((HOST, PORT))
    rospy.init_node('visual_grounding_client', anonymous=False, disable_signals=True)

    agent = Agent()
    idx = 500
    rospy.loginfo('Clear Table to Get Placement Info')
    agent.arm.to_camera_pose()
    input('Press any key when table is clean ...')
    clean_cloud = agent.camera.get_cloud()
    while not rospy.is_shutdown():
        try:
            rospy.loginfo('Visual Grounding Trial {}'.format(idx+1))
            rospy.loginfo('To Camera Pose')
            agent.arm.to_camera_pose()
            rospy.sleep(2.0)
            input('Place Things and Get Visual Info (Press any key)')
            # Take visual info and send image to the server
            cv_img = agent.camera.get_rgb()
            cloud = agent.camera.get_cloud()
            # Send Captured Image
            retval, buf = cv2.imencode('.png', cv_img, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT])
            data = np.array(buf)
            b64_data = base64.b64encode(data)
            length = str(len(b64_data))
            cli_sock.sendall(length.encode('utf-8').ljust(64))
            cli_sock.sendall(b64_data)
            rospy.loginfo('Wait for the server')
            # Receive bounding box Info
            data = cli_sock.recv(1024)
            str_data = data.decode()
            bbox = str_data.strip().split(';')
            TL_X = int(bbox[0])
            TL_Y = int(bbox[1])
            BR_X = int(bbox[2])
            BR_Y = int(bbox[3])
            PL_X = int(bbox[4])
            PL_Y = int(bbox[5])
            # Visualize received bbox on input image
            cv_img = cv2.rectangle(cv_img, (TL_X, TL_Y), (BR_X, BR_Y), (0, 0, 255))
            cv_img = cv2.circle(cv_img, (PL_X, PL_Y), 8, (0, 255, 0), -1)
            cv2.imwrite('result/vg_{}.png'.format(idx), cv_img)
            rospy.loginfo('Pick bbox: {} {} {} {}'.format(TL_X, TL_Y, BR_X, BR_Y))
            rospy.loginfo('Place coordinate: {} {}'.format(PL_X, PL_Y))
            # Manipulation (Pick-n-Place)
            agent.arm.to_ready_pose()
            agent.pick_n_place(TL_X, TL_Y, BR_X, BR_Y, PL_X, PL_Y, cloud, clean_cloud)
            idx += 1
            value = input('Continue? [y/n]')
            if value != 'y': break
        except KeyboardInterrupt:
            print('\nClient Ctrl-c')
            break
        except IOError and ValueError:
            print('\nServer closed')
            break

    cli_sock.close() 
    return

if __name__ == '__main__':

    main()






