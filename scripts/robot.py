
# from gps import coords_callback
from multiprocessing.dummy import Value
from uvc import UVC
#from gps import get_coords
import time
from datetime import datetime
import sys
import csv

from utils import *
#from log_uvc import log_data
import numpy as np
from utils import to_cartesian
norm = np.linalg.norm # this might not be right


K_RHO = 1
K_ALPHA = -75

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
        #waypoints = np.array([[34.1064, -117.7125], [34.1063, -117.7125]])
        waypoints = np.array([[34.106195, -117.712030], [34.106168, -117.712045]])
        self._origin = waypoints[0]
        self._waypoints = [to_cartesian(waypoint, waypoints[0]) for waypoint in waypoints]
        self._thrust_control = 0
        self._yaw_control = 0
        self._gps = ('','')
        self._heading = 0

    def _get_controls(self, heading, coords, waypoint, uvc):
        robot_to_waypoint = waypoint - coords
        heading = heading - 90
        robot_vec = np.array([np.cos(heading*(np.pi/180)), np.sin(heading*(np.pi/180))])
        print("ROBOT to waypoint", robot_to_waypoint)
        print("ROBOT BEC", robot_vec)
        err_rho = norm(robot_to_waypoint)
        err_angle = angle_between(robot_to_waypoint, robot_vec)
        print("error angle", err_angle*180/np.pi)

        #thrust_control = int(min(max(K_RHO * err_rho + 128, 0), 255))
        yaw_control = int(min(max(K_ALPHA * err_angle + 128, 1), 255))
        print("integer yaw control here")
        print(yaw_control)
        thrust_control = 128
        
        thrust_control = uvc.to_hex(thrust_control)
        yaw_control = uvc.to_hex(yaw_control)

        return [thrust_control, yaw_control]

    # Define a callback to log data
    def _log_data(self, uvc):
        knots_per_meter = 1.944
        latitude, longitude = uvc.get_coords(default=('',''))
        x_speed, y_speed = uvc.get_speeds(default=(np.nan, np.nan))
        if np.isnan(x_speed) or np.isnan(y_speed):
            speed = ''
        else:
            print("Logdata variables")
            speed = np.sqrt(x_speed**2 + y_speed**2) * knots_per_meter
            print(latitude, longitude, speed, uvc.get_heading(default=''))

        print("GPS", type(self._gps[0]))

        if self._gps[0] != '':
            gps_x = to_cartesian(np.array([latitude,longitude]), self._origin)[0]
            gps_y = to_cartesian(np.array([latitude,longitude]), self._origin)[1]
        else:
            gps_x = 0
            gps_y = 0

        if uvc.get_heading(default='') == '':
            robot_vector = [0,0]
        else:
            robot_vector = [np.cos(uvc.get_heading(default='')*(3.14159/180)), np.sin(uvc.get_heading(default='')*(3.14159/180))]
            
        data = [
            datetime.now(),
            latitude,
            longitude,
            robot_vector,
            gps_x,
            gps_y,
            self._thrust_control,
            self._yaw_control,
            speed,
            uvc.get_heading(default='')
        ]

        # Write to a savefile if one was given and to the console
        if savefile is not None:
            writer.writerow(data)
        print(','.join([str(datum) for datum in data]))
        return data

    def _track_waypoint(self, uvc_object, waypoint):

        current_data = self._log_data(uvc_object)
        #print("current data is")
        #print(current_data)
        
        lat, lon = current_data[1:3]
        #lat = 34.106046
        #lon = -117.711964

        # it reaches the default state
        if (lat, lon) != ('',''):
            print("got gps")
            self._gps = to_cartesian(np.array([lat, lon]), self._origin)
            
        print("GPS X", self._gps[0])
        print("GPS Y", self._gps[1])

        heading = current_data[-1]
        if heading != None:
            print("got heading", self._heading)
            self._heading = heading

        if self._gps[0] != '' and heading != '':
            
            print("COMPASS", heading)
            print(waypoint)
            print(np.asarray(self._gps))
            [thrust, yaw_angle] = self._get_controls(self._heading, np.asarray(self._gps), waypoint, uvc_object)
            #[thrust, yaw_angle] = self._get_controls(20, np.array([0, 1]), np.array([1,2]), uvc_object)
            self._thrust_control = thrust
            self._yaw_control = yaw_angle
            print("Thrust P control", thrust)
            print("Yaw angle", yaw_angle)

            

            # send waypoint parameters
            output = uvc_object._write_command('OMP','{}{}8080{}'.format(yaw_angle, yaw_angle, thrust), '00', '10')
            return output
        return None

if __name__ == '__main__':

    # Read in command line arguments: savepath to a file to dump data
    _, *rest = sys.argv
    
    # Define the column names of the data to be logged
    columns = [
                'datetime',
                'Latitude',
                'Longitude',
                'Cartesian X',
                'Cartesian Y',
                'Robot Vector',
                'Thrust Control', 
                'Yaw Control',
                'Vehicle Speed (Kn)',
                'C True Heading'
            ]

    # Open a file to save the data to if a savepath was given
    if len(rest) != 0:
        savepath = rest[0]
        directory, filename = os.path.split(savepath)
        mkdir(directory)
        savefile = open(add_version(savepath), 'w', newline='')

        # Write the csv header
        writer = csv.writer(savefile)
        writer.writerow(columns)
    else:
        savepath = None
        savefile = None
    
    uvc = UVC('COM1', verbosity=1)

    def request_data(uvc):
        # request sensor data
        uvc._write_command('OSD','C','G','S','P','Y','D','T','I')

    #uvc.on_listening_run(request_data)
    #uvc.on_listening_run(log_data(uvc))
    
    uvc.start()

    robot = Robot()

    next_waypoint = robot._waypoints[1]

    print(next_waypoint)
    print(','.join(columns))
    while uvc.is_listening():
        uvc._write_command('OSD','C','G','S','P','Y','D','T','I')
        value = robot._track_waypoint(uvc, next_waypoint)
        print("Command wrote to track waypoint", value)

        uvc.run()
        time.sleep(1)

    if not uvc.is_closed():
        # Stop UVC
        uvc.stop()
        uvc.close()
        
    # Close savefile if one was created
    if savefile is not None:
        savefile.close()
