"""Utility methods used for determining what is in a vehicle's field of view."""
import math
import matplotlib.pyplot as plt

def observed(position, orientation, target_position, fov=180, looking_distance=50):
    """Check if a single vehicle/pedestrian can see another vehicle/pedestrian.

    Parameters
    ----------
    position : tuple of (float, float)
        (x, y) position of observer
    orientation : float
        angle of observer in degrees (0 deg is East, 90 deg is North)
    target_position: tuple of (float, float)
        (x, y) position of the target
    fov: float
        the field of view of the observer
    looking_distance: float
        how far the observer can see

    Return: boolean
        whether or not the target can be observed
    """
    delta_x = target_position[0] - position[0]
    delta_y = target_position[1] - position[1]

    # edge case where both objects are at the same position
    if delta_x == 0 and delta_y == 0:
        return True

    # object is too far
    if euclidian_distance(delta_x, delta_y) > looking_distance:
        return False

    angle = get_angle(delta_x, delta_y)
    right_angle = (orientation - angle) % 360
    left_angle = (angle - orientation) % 360

    # object is not in FOV
    if left_angle > fov/2.0 and right_angle > fov/2.0:
        return False
    
    return True

def orientation_unit_circle(angle):
    """Convert SUMO's angle to the standard unit circle.

    SUMO defines 0 degrees facing North with the angle increasing clockwise (90 is East,
    180 is South, etc.). This method converts SUMO's defintion to the standard unit circle
    where 0 is East, 90 is North, etc.

    Parameters
    ----------
    angle : float
        SUMO's angle value

    Return: float
        the angle represented in the standard unit circle
    """
    return (360 - (angle - 90)) % 360

def euclidian_distance(x, y):
    """Get euclidian distance between two segments.

    Parameters
    ----------
    x : float
        length of the first edge
    y : float
        length of the second edge

    Return: float
        euclidian distance between the two edges
    """
    return math.sqrt(x**2 + y**2)

def get_angle(x, y):
    """Get angle based on the unit circle.

    Parameters
    ----------
    x : float
        x-value
    y : float
        y-value

    Return: float
        angle
    """
    if x == 0:
        if y > 0:
            return 90
        else:
            return 270
    elif x < 0:
        return math.degrees(math.atan(y / x)) + 180

    return math.degrees(math.atan(y / x))

def get_blocked_segments(position, target_position, target_orientation, target_length, target_width):
    """Define a line segment that blocks the observation vehicle's line of sight.

    From the perspective of the observation vehicle, define the longest line segment between
    the four corners of the observed vehicle that blocks the widest field of view. This is
    done by selecting the two x, y points that create the largest angle with respect to the
    position of the observation vehicle.

    Parameters
    ----------
    position : tuple
        x, y position of the observation vehicle
    target_position : tuple of (float, float)
        x, y position of the vehicle being observed
    target_length : float
        length of the observed vehicle
    target_width : float
        width of the observed vehicle

    Return: tuple of (tuple, tuple)
        Each element is another length 2 tuple of the x and y positions of the line segment
        that blocks the observation vehicle's field of view
    """
    corner_angle = math.degrees(math.atan(target_width / target_length))
    corner_dist = euclidian_distance(target_length / 2, target_width / 2)

    corners = get_corners(target_position[0], target_position[1], target_orientation, \
            corner_angle, corner_dist)

    angles = []
    for i, c in enumerate(corners):
        angles.append((i, get_angle(position[0] - c[0], position[1] - c[1])))

    max_angle = corners[max(angles, key=lambda x: x[1])[0]]
    min_angle = corners[min(angles, key=lambda x: x[1])[0]]

    return(max_angle, min_angle)

def get_corners(x, y, orientation, corner_angle, corner_dist, center_offset=1.5):
    corners = []

    adjusted_x = x - center_offset * math.cos(math.radians(orientation))
    adjusted_y = y - center_offset * math.sin(math.radians(orientation))

    t_angle = math.radians((orientation + corner_angle) % 360)
    corners.append((adjusted_x + math.cos(t_angle) * corner_dist, \
            adjusted_y + math.sin(t_angle) * corner_dist))

    t_angle = math.radians((orientation + 180 - corner_angle) % 360)
    corners.append((adjusted_x + math.cos(t_angle) * corner_dist, \
            adjusted_y + math.sin(t_angle) * corner_dist))

    t_angle = math.radians((orientation + 180 + corner_angle) % 360)
    corners.append((adjusted_x + math.cos(t_angle) * corner_dist, \
            adjusted_y + math.sin(t_angle) * corner_dist))

    t_angle = math.radians((orientation - corner_angle) % 360)
    corners.append((adjusted_x + math.cos(t_angle) * corner_dist, \
            adjusted_y + math.sin(t_angle) * corner_dist))

    return corners

def check_blocked(position, target_position, blocked, vehicle_id):
    """Check if a target vehicle is blocked by another vehicle or object.

    Create a line-of-sight line segment between the observation and target
    vehicles. Return True if this line intersects with a line segment in blocked
    that corresponds to a different vehicle_id; otherwise return False.

    Parameters
    ----------
    position : tuple
        x, y position of the observation vehicle
    target_position : tuple
        x, y position of the target vehicle
    blocked : dict of {str : tuple}
        key : unqie identifier for vehicle that is blocking the view
        tuple of (tuple, tuple) : two tuples define the x and y positions of the
            endpoints of a line segment that blocks field of view
    vehicle_id : str
        unique identifier for the observation vehicle

    Return: boolean
        whether or not a vehicle is blocked from the observation vehicle's point of view
    """
    for b in list(blocked):
        if b == vehicle_id:
            continue
        line_of_sight = (position, target_position)
        if lines_intersect(line_of_sight, blocked[b]):
            return True
    return False

def lines_intersect(line1, line2):
    """Check if two lines intersect.

    Algorithm defined here:
    https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    Parameters
    ----------
    line1 : tuple of (tuple, tuple)
        each element is the x, y position defining an endpoint of line1
    line2 : tuple of (tuple, tuple)
        each element is the x, y position defining an endpoint of line2

    Return: boolean
        whether or not the two lines intersect
    """
    def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

    a, b, c, d = line1[0], line1[1], line2[0], line2[1]
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def visualize_vision(observation_vehicle, block_lines, viewed_veh, viewed_ped):

    plt.axes()
    plt.clf()

    for v_id in viewed_veh:
        x, y = viewed_veh[v_id]['xy']
        yaw = viewed_veh[v_id]['yaw']
        width = viewed_veh[v_id]['width']
        length = viewed_veh[v_id]['length']
        if viewed_veh[v_id]['viewed']:
            color = 'y'
        else:
            color = 'r'

        corner_angle = math.degrees(math.atan(width / length))
        corner_dist = euclidian_distance(length / 2, width / 2)
        corners = get_corners(x, y, yaw, corner_angle, corner_dist)

        # plot vehicle perimeter
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            x_values = [a[0], b[0]]
            y_values = [a[1], b[1]]
            plt.plot(x_values, y_values, color)

        # plot vehicle position (SUMO defines position to be center of hood)
        plt.scatter(x, y, marker='s', s=2)

    for ped_id in viewed_ped:
        x, y = viewed_ped[ped_id]['xy']
        plt.scatter(x, y, marker='*', c='b')

    # plot blocking lines
    for line in block_lines.values():
        a, b = line[0], line[1]
        x_values = [a[0], b[0]]
        y_values = [a[1], b[1]]
        plt.plot(x_values, y_values, 'k')

    plt.axis('equal')
    axes = plt.gca()
    x, y = observation_vehicle['xy']
    axes.set_xlim([x - 50, x + 50])
    axes.set_ylim([y - 50, y + 50])

    plt.draw()
    plt.pause(0.001)
