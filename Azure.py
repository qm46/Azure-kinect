import pyk4a as k4a
import numpy as np
import cv2
import time
import pickle

# Initialize Azure Kinect sensor
config = k4a.Config(
    color_resolution=k4a.ColorResolution.RES_720P,
    depth_mode=k4a.DepthMode.WFOV_UNBINNED,
    synchronized_images_only=True,
    camera_fps=k4a.FPS.FPS_15,
)


# Initialize accelerometer and gyroscope data
acc_data = np.zeros((3,))
gyro_data = np.zeros((3,))
prev_time = time.time()

i=0

# Initialize camera pose
pose = np.eye(4)
kinect = k4a.PyK4A(config)

# Start camera capture
kinect.start()


# Get calibration data
calibration = kinect.calibration

while True:
    time.sleep(0.7)
    # Get the next capture from the Azure Kinect
    capture = kinect.get_capture(-1)

    # Extract RGB image from the capture
    color_image = capture.color
    color_image_np = np.array(color_image)

    # Save RGB image to file
    cv2.imwrite(f"./rgb/color_{i}.png", color_image_np)

    img = color_image_np

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the orange color in HSV(only for wood cube)
    lower_orange = np.array([8, 50, 50])
    upper_orange = np.array([11, 255, 255])

    # Create a mask using the color threshold
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = np.where(mask == 255, 1, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw a bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    result = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Create a new mask that is the same size as the original image and fill it with zeros
    #new_mask = np.zeros_like(mask)

    # Set the values inside the bounding box to be the same as the values in the original mask
    #new_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w]

    #mask = np.zeros((new_mask.shape[0],new_mask.shape[1],3), dtype=int)

    #mask[:,:,0] = new_mask
    #mask[:,:,1] = new_mask
    #mask[:,:,2] = new_mask

    #cv2.imwrite(f"./mask/mask_{i}.png", mask)

    

    #bbox = np.array([x, y, w, h])
    #np.savetxt(f"./bbox/bbox_{i}.txt", bbox.reshape(1, -1), delimiter="\t", fmt="%d")

    # Extract point cloud from the capture
    depth_map = capture.transformed_depth  
    depth_map_np = np.array(depth_map)
    cv2.imwrite(f"./depth_map/depth_map_{i}.png", depth_map)

    #point_cloud = capture.depth_point_cloud
    #point_cloud_np = np.array(point_cloud)
    #point_cloud_np = point_cloud_np.reshape((point_cloud_np.shape[0] * point_cloud_np.shape[1], 3))
    #with open(f"./point_cloud/point_cloud_{i}.pkl", "wb") as f:
    #    pickle.dump(point_cloud_np, f)


    #np.savetxt(f"point_cloud_{i}.txt", point_cloud.reshape((point_cloud.shape[0] * point_cloud.shape[1], 3)), delimiter=",")


    # Extract intrinsic matrix from the calibration data for the depth camera
    intrinsicsd = calibration.get_camera_matrix(k4a.calibration.CalibrationType.DEPTH).reshape(3, 3)

    # Save intrinsic matrix to file
    np.savetxt(f"./intrinsic_depth/intrinsicd_{i}.txt", intrinsicsd, delimiter=" ")

    intrinsicsc = calibration.get_camera_matrix(k4a.calibration.CalibrationType.COLOR).reshape(3, 3)

    np.savetxt(f"./intrinsic_rgb/intrinsicc_{i}.txt", intrinsicsc, delimiter=" ")

    # imu_sample = kinect.get_imu_sample()
    # if imu_sample:
    #     # Extract accelerometer and gyroscope data
    #     acc_data = np.array(imu_sample["acc_sample"])
    #     gyro_data = np.array(imu_sample["gyro_sample"])
        
    #     # Calculate time difference
    #     current_time = time.time()
    #     time_diff = current_time - prev_time
    #     prev_time = current_time
        
    #     # Calculate rotation matrix from gyroscope data
    #     omega = np.radians(gyro_data) / time_diff
    #     theta = np.linalg.norm(omega)
    #     if theta > 0:
    #         omega_hat = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    #         R = np.eye(3) + np.sin(theta) * omega_hat / theta + (1 - np.cos(theta)) * np.dot(omega_hat, omega_hat) / (theta ** 2)
    #     else:
    #         R = np.eye(3)
        
    #     # Update camera pose
    #     pose[:3, :3] = np.dot(pose[:3, :3], R)
    #     pose[:3, 3] += time_diff * pose[:3, :3].dot(acc_data) 
    #     #pose[:3, :3] = np.eye(3)
    #     #pose[:3, 3] = np.transpose([1,1,1])
    # # Save camera pose to file
    # np.savetxt(f"./extrinsic/pose_{i}.txt", pose[:3, :4], delimiter=" ")
    

    i = i+1
    print(i)

    # Wait for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop camera capture
kinect.stop()
