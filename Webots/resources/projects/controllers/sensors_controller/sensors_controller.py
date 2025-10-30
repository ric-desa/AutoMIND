from controller import Robot
import numpy as np
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

cam_front = robot.getDevice("camera_front")   # camera name in the world must be "camera"
cam_back = robot.getDevice("camera_back")
cam_right = robot.getDevice("camera_right")

lidar = robot.getDevice("lidar")
lidar.enable(timestep)
lidar.enablePointCloud()

cam_front.enable(timestep)
cam_back.enable(timestep)
cam_right.enable(timestep)


# query geometry/info
h_res = lidar.getHorizontalResolution()
layers = lidar.getNumberOfLayers()    # usually 1 for Hokuyo
fov = lidar.getFov()                  # horizontal field of view in radians
max_range = lidar.getMaxRange()

print("Lidar info: horiz_res =", h_res, "layers =", layers,
      "fov(deg) =", math.degrees(fov), "max_range =", max_range)



while robot.step(timestep) != -1:
    image = cam.getImage()
#     # process or leave it running

    ranges = lidar.getRangeImage()  # returns a flat list of floats
    if ranges is None:
        print("No range image (maybe device not enabled or simulation stopped)")
    else:
        # reshape into (layers, h_res)
        arr = np.array(ranges, dtype=np.float32)
        arr = arr.reshape((layers, h_res))
        print("range[0..5] of layer0:", arr[0, :6])

    # 2) Point cloud (list of points in lidar frame)
    pts = lidar.getPointCloud()  # returns list of point objects (x,y,z)
    if pts is not None:
        print("point cloud size:", len(pts))
        if len(pts) > 0:
            # print the first point
            p = pts[0]
            print("first point:", p.x, p.y, p.z)

