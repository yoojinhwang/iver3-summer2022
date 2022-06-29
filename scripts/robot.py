# from gps import coords_callback
from multiprocessing.dummy import Value
from uvc import UVC
from gps import get_coords
import time

from utils import *
from log_uvc import log_data
import numpy as np
from utils import to_cartesian
norm = np.linalg.norm # this might not be right

K_RHO = 1
K_ALPHA = 1

class Robot():
    '''
    Robot state machine.

    States
    ------
    create uvc object
    waypoints? OJW (jump to waypoint)

    in the end we want thrust, pitch, and heading commands (send to UVC)

    functions: 
    - inputting coordinates (waypoints to go to) within a mission
    - p control to move to the desired point after collecting GPS and compass data
        - get thrust
        - get yaw
    
    states: 
    - track waypoint
    - park
    - collection: to get the gps and compass data
    - move robot's thrusters from input of the collection and the waypoint
    get coords and get heading for compass that UVC gave to the backseat
    '''
    def __init__(self):

        # log compass and gps data at the same time
        self._waypoints = np.array([(34.1064, -117.7125), (34.1063, -117.7125)])
        self._gps = ('','')
        self._heading = 0

    def _get_controls(heading, coords, waypoint):
        robot_to_waypoint = waypoint - coords
        robot_vec = np.array([np.cos(heading), np.sin(heading)])
        err_rho = norm(robot_to_waypoint)
        err_angle = angle_between(robot_to_waypoint, robot_vec)
        thrust_control = K_RHO * err_rho
        yaw_control = K_ALPHA * err_angle

        return [thrust_control, yaw_control]

    def _track_waypoint(self, uvc_object, origin, waypoint): 
        current_data = log_data()
        speed = current_data[3]
        [lat, lon] = current_data[1:2]

        # it reaches the default state
        if (lat, lon) != ('',''):
            print("got gps")
            self._gps = self.to_cartesian((lat, lon), origin)
            
        print("GPS X", self._gps[0])
        print("GPS Y", self._gps[1])

        heading = current_data[4]
        if heading != None:
            self._heading = heading
            
        print("COMPASS", heading)
        
        [thrust, yaw_angle] = self._get_controls(self._heading, np.array([self._gps[0], self._gps[1]]), waypoint)
        
        print("Thrust P control", thrust)
        print("Yaw angle", yaw_angle)

        # send waypoint parameters
        output = uvc_object._write_command('OMP','{}{}'.format(yaw_angle, yaw_angle), '80','80', '{}'.format(thrust), '00', '10')

        return output
        
if __name__ == '__main__': 
    uvc = UVC('COM1', verbosity=1)
    uvc.start()

    robot = Robot()

    origin = robot._waypoints[0] # Nnn.nnnnn, Nnn.nnnnn
    next_waypoint = robot._waypoints[1]

    while uvc.is_listening(): 

        # request sensor data
        uvc._write_command('OSD','C','G','S','P','Y','D','T','I')

        value = robot._track_waypoint(uvc, origin, next_waypoint)
        print("Command wrote to track waypoint", value)

        uvc.run()
        time.sleep(1)
        
    
