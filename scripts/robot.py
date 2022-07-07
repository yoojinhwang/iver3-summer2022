from uvc import UVC
import time
from datetime import datetime
import sys
import csv
from utils import *
import numpy as np
from utils import to_cartesian
norm = np.linalg.norm

K_RHO = 6
K_ALPHA = 75
K_DEPTH = 1
K_ROLL = 1
DIST_THRESH = 5
SEND_RATE = 0.5

class Robot():
    '''
    Robot class.
    '''
    def __init__(self):

        # log compass and gps data at the same time
        #waypoints = np.array([[34.1064, -117.7125], [34.1063, -117.7125]])
        #waypoints = np.array([[34.106195, -117.712030], [34.106168, -117.712045]])
        #waypoints = np.array([[34.106196, -117.712059],
        #                      [34.106203, -117.711912],
        #                      [34.105996, -117.711960],
        #                      [34.106013, -117.712048]])
        # single line
        waypoints = np.array([[34.109191, -117.712723],
                              [34.109096, -117.712535],
                              [34.109191, -117.712723]])

        waypoint_depths = np.array([0, 0, 0])
        # triangle
        #waypoints = np.array([[34.109191, -117.712723],
        #                      [34.109040, -117.712552],
        #                      [34.109150, -117.712503],
        #                      [34.109191, -117.712723]])
        
        self._origin = waypoints[0]
        self._waypoints = [to_cartesian(waypoint, waypoints[0]) for waypoint in waypoints]
        self._waypoint_depths = waypoint_depths
        self._waypoint_index = 0
        self._current_waypoint = self._waypoints[0]
        self._thrust_control = 0
        self._yaw_control = 0
        self._dist_to_waypoint = 100
        self._latlon = np.array([0, 0])
        self._gps = np.array([0,0])
        self._speed = 0
        self._heading = 0
        self._pitch = 0
        self._roll = 0

    def _get_controls(self):
        '''Gets controls to go to the next waypoint'''
        index = self._waypoint_index
        waypoint = self._waypoints[index]
        des_depth = self._waypoint_depths[index]
        heading = self._heading
        robot_to_waypoint = waypoint - self._gps
        heading = np.radians(-heading + 90)
        robot_vec = np.array([np.cos(heading), np.sin(heading)])

        print("ROBOT to waypoint", robot_to_waypoint)
        print("ROBOT BEC", robot_vec)

        err_rho = norm(robot_to_waypoint)
        err_angle = angle_between(robot_to_waypoint, robot_vec)
        err_depth = des_depth - self._depth

        self._dist_to_waypoint = err_rho
        print("error angle", err_angle*180/np.pi)

        thrust_control = int(min(max(K_RHO * err_rho + 128, 0), 255))
        yaw_control = int(min(max(K_ALPHA * err_angle + 128, 1), 255))
        left_control = int(min(max(K_DEPTH * err_depth + K_ROLL * self._roll + 128, 1), 255))
        right_control = int(min(max(K_DEPTH * err_depth - K_ROLL * self._roll + 128, 1), 255))

        print("integer yaw control", yaw_control)
        #thrust_control = 128

        return [thrust_control, yaw_control, left_control, right_control]

    # Define a callback to log data
    def _update_states(self, uvc):
        '''Update robot states and log data to file'''
        knots_per_meter = 1.944

        # get current states from uvc
        lat, lon = uvc.get_coords(default=('',''))
        x_speed, y_speed = uvc.get_speeds(default=('', ''))
        heading = uvc.get_heading(default='')
        pitch = uvc.get_pitch(default='')
        roll = uvc.get_roll(default='')

        uvc_data = [lat, lon, x_speed, y_speed, heading, pitch, roll]
        all_data_present = not any(datum == '' for datum in uvc_data)

        # update robot states
        if all_data_present:
            print("got data")
            speed = np.sqrt(x_speed**2 + y_speed**2) * knots_per_meter
            coords_cart = to_cartesian(np.array([lat, lon]), self._origin)
            self._gps = np.asarray(coords_cart)
            self._heading = heading
            self._pitch = pitch
            self._roll = roll

            # print for debugging
            print("raw gps", lat, lon)
            print("cartesian", coords_cart)
            print("angles", heading, pitch, roll)
            print("speed", speed)


    def _log_data(self):
        '''Logs the current state of the robot to file, if applicable'''

        # calculations for logging
        transformed_heading = np.radians(-self._heading + 90)
        robot_vector = [np.cos(transformed_heading), np.sin(transformed_heading)]
        robot_to_waypoint = self._current_waypoint - self._gps
        
        # data to log
        data = [
            datetime.now(),
            self._latlon[0],
            self._latlon[1],
            robot_vector,
            robot_to_waypoint,
            self._gps[0],
            self._gps[1],
            self._thrust_control,
            self._yaw_control,
            self._speed,
            self._heading]

        # Write to a savefile if one was given and to the console
        if savefile is not None:
            writer.writerow(data)
        print(','.join([str(datum) for datum in data]))
        return data

    def _track_waypoint(self, uvc_object):
        '''Updates state, logs current data, and sends controls to move the robot'''
        
        self._update_states(uvc_object)
        self._log_data()

        robot_data = [self._gps[0], self._gps[1], self._heading, self._pitch, self._roll]
        all_data_present = not any(datum == 0 for datum in robot_data)

        if all_data_present:

            [thrust, yaw_angle, left_angle, right_angle] = self._get_controls()
            thrust = uvc.to_hex(thrust)
            yaw_angle = uvc.to_hex(yaw_angle)
            left_angle = uvc.to_hex(left_angle)
            right_angle = uvc.to_hex(right_angle)

            self._thrust_control = thrust
            self._yaw_control = yaw_angle
            print("Thrust P control", thrust)
            print("Yaw angle", yaw_angle)

            # send waypoint parameters
            output = uvc_object._write_command('OMP','{}{}{}{}{}'.format(yaw_angle, yaw_angle, left_angle, right_angle, thrust), '00', '10')
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
                'Robot to Waypoint Vector',
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
    uvc.start()
    robot = Robot()

    print(','.join(columns))

    start_time = time.time()

    while robot._waypoint_index < len(robot._waypoints):
        waypoint = robot._waypoints[robot._waypoint_index]
        robot._current_waypoint = waypoint
        print("waypoint index", robot._waypoint_index)
        print("waypoint coordinate", waypoint)
        if uvc.is_listening():
            uvc._write_command('OSD','C','G','S','P','Y','D','T','I') # request data
            value = robot._track_waypoint(uvc)
            print("Command wrote to track waypoint", value)

            uvc.run()
            
            passed_time = time.time() - start_time
            time.sleep(SEND_RATE - (passed_time % SEND_RATE)) # send commands at a constant rate

        if robot._dist_to_waypoint < DIST_THRESH: # move on to next waypoint
            print(robot._dist_to_waypoint, "close enough")
            robot._waypoint_index += 1

    if not uvc.is_closed():
        # Stop UVC
        uvc.stop()
        uvc.close()
        
    # Close savefile if one was created
    if savefile is not None:
        savefile.close()
