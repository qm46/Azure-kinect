import numpy as np
import open3d as o3d
import json
import pickle
import os

def create_point_cloud_from_rgbd_image(color, depth, intrinsic, extrinsic, mask):
    color = color * mask[:, :, np.newaxis]
    depth = depth * mask

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        depth_scale=1000,
        depth_trunc=1.0,
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic,
        extrinsic
    )
    return pcd

if __name__ == "__main__":

    dataset_path = "./dataset"
    rgb_dir = "rgb"

    dir_ = os.path.join(dataset_path, rgb_dir)
    total_files = len([f for f in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, f))])
    count = 0

    while(count < total_files):

        # Load the color and depth images
        color = o3d.io.read_image(f"./dataset/rgb/{count}.png")
        depth = o3d.io.read_image(f"./dataset/depth/{count}.png")
        mask = o3d.io.read_image(f"./dataset/mask/{count}.png")
        mask = np.asarray(mask)
        #mask[:] = 1
        mask = mask.astype(bool)

        pose = np.load(f'./dataset/pose/{count}.npy')
        R = pose[0:3, 0:3]
        T = pose[0:3, 3]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T

        with open('./dataset/intrinsics.json') as f:
            intrinsics = json.load(f)
        f_x = intrinsics['fx']
        f_y = intrinsics['fy']
        c_x = intrinsics['ppx']
        c_y = intrinsics['ppy']


        # Load the camera intrinsic parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=1280,
            height=720,
            fx=f_x,
            fy=f_y,
            cx=c_x,
            cy=c_y
        )

        # Create the point cloud
        pcd = create_point_cloud_from_rgbd_image(color, depth, intrinsic, extrinsic, mask)

        # Visualize the point cloud
        #o3d.visualization.draw_geometries([pcd])
        if not os.path.exists("./dataset/point_cloud/"):
            os.makedirs("./dataset/point_cloud/")
        o3d.io.write_point_cloud(f"./dataset/point_cloud/{count}.ply", pcd)

        points = np.asarray(pcd.points)

        # Load the camera intrinsic and extrinsic parameters
        intrinsics_matrix = np.array([[f_x, 0, c_x],
                                      [0, f_y, c_y],
                                      [0, 0, 1]])

        extrinsics_matrix = np.hstack((R, np.expand_dims(T, -1)))

        # Initialize a dictionary to store the correspondences
        correspondences = {}

        # Iterate over all 3D points and establish correspondences
        for point_3d in points:
            # Transform the 3D point to camera coordinates
            point_3d_hom = np.hstack((point_3d, 1))
            point_camera_hom = extrinsics_matrix @ point_3d_hom
            point_camera = point_camera_hom[:3]

            # Project the 3D point onto the camera image plane
            point_hom = intrinsics_matrix @ point_camera
            point_2d_hom = point_hom / point_hom[2]
            point_2d = point_2d_hom[:2]

            # Store the correspondence between the 3D point and the corresponding 2D point
            correspondences[tuple(point_3d)] = tuple(point_2d)


        string_3d = []
        string_2d = []
        for key, val in correspondences.items():
            string_3d.append(key)
            string_2d.append(val)
        string_3d = np.array(string_3d)
        string_2d = np.array(string_2d)

        points_3d_dir = "./dataset/3d_array/"
        points_2d_dir = "./dataset/2d_array/"

        if not os.path.exists(points_3d_dir):
            os.makedirs(points_3d_dir)
        if not os.path.exists(points_2d_dir):
            os.makedirs(points_2d_dir)

        np.savetxt(f'{points_3d_dir}/{count}.txt', string_3d)
        np.savetxt(f'{points_2d_dir}/{count}.txt', string_2d)

        with open(f"{points_3d_dir}/{count}.pkl", 'wb') as f:
            pickle.dump(string_3d, f)

        with open(f"{points_2d_dir}/{count}.pkl", 'wb') as f:
            pickle.dump(string_2d, f)
        count += 1

