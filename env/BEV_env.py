import carla
import os
from os.path import join
import random
import numpy as np
from actor.vehicle import bounding_box
from sensor.rgb_camera import turn_world_cood_to_camera_coord, \
                                get_intrinsic_matrix,\
                                create_surrond_camera_trans, \
                                turn_camera_cood_to_world_coord
from sensor.fisheye_camera import create_fisheye_camera_trans
from utils.geometry import turn_trans_to_list, \
                            turn_extent_to_list, \
                            turn_velocity_to_speed, \
                            get_map_center
import cv2
import time
from copy import deepcopy
import yaml

    # 定义语义分割相机回调函数
def semantic_camera_callback(data, ego_vehicle_id):
    # 获取语义分割图像数据
    semantic_data = np.frombuffer(data.raw_data, dtype=np.uint8)
    semantic_data = np.reshape(semantic_data, (data.height, data.width, 4))[:,:,:3]
    semantic_data = semantic_data[:,:,::-1] * 30
    cv2.imwrite(join("/mnt/pool1/carla/output", "vehicle_{}_{}.png".format(ego_vehicle_id, data.frame)), semantic_data)


class Env():
    def __init__(self, 
                       env_name ,
                       ip = "localhost",
                       port = 2000,
                       map_name = "Town01",
                       time_step = 0.05,
                       cav_num = 3,
                       npc_num = 3,
                       bev_shape = [],
                       save_path = "/mnt/pool1/carla/output",
                       use_fisheye = False,
                       cav_trans = [],
                       neighbor_trans = [],
                       use_autopilot = []
                       ) -> None:
        
        # Connect to the client and retrieve the world object
        self.client = carla.Client(ip, port)
        self.client.load_world(map_name)

        traffic_manager = self.client.get_trafficmanager()
        # tm里的每一辆车都要和前车保持至少3m的距离来保持安全
        # traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        # tm里面的每一辆车都是混合物理模式
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_synchronous_mode(True)

        self.cav_num = cav_num
        self.npc_num = npc_num

        self.cav_ids = []

        # vehicles
        self.vehicles = {}

        # rgb cameras
        self.rgb_cameras = {}

        # semantic cameras
        self.semantic_cameras = {}

        # svc_cameras
        self.svc_cameras = {}
        self.svc_depth_cameras = {}
        self.svc_sem_cameras = {}

        # svc_depth, semgemenmtaion result
        self.svc_depth = {}
        self.svc_vehicle_mask = {}

        # fisheye cameras
        self.fisheye_cameras = {}

        # BEV_tran
        self.bev_trans = carla.Transform(carla.Location(z = 20), carla.Rotation(pitch = -90))
        self.bev_cam_trans = {}

        # Load world
        self.world = self.client.get_world()

        # Set update time interval and synchronous_mode 
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = time_step #20 fps, 5ms
        settings.max_substeps = 5
        settings.max_substep_delta_time = time_step / 5  # fixed_delta_seconds
        self.world.apply_settings(settings)

        self.spawn_points = self.world.get_map().get_spawn_points()
        print("Number of All Spawn Points is {}".format(len(self.spawn_points)))

        # all_drivable_lanes = self.world.get_map().get_all_landmarks_of_type()
        self.center_point = get_map_center(self.world.get_map())
        
        # sample spawn points from a given point within certain range
        scale_range = 20
        self.spawn_points = [point for point in self.spawn_points \
                                if np.sqrt((point.location.x-self.center_point[0])**2 + \
                                        (point.location.y - self.center_point[1])**2) < scale_range]
        points = np.array([(point.location.x, point.location.y) for point in self.spawn_points])
        np.set_printoptions(precision=1, suppress=True)
        print(np.sqrt(np.sum((points[None,...]-points[:,None, :])**2, axis = 2)))
        print("Number of Remain Spawn Points is {}".format(len(self.spawn_points)))

        self.bp_lib = self.world.get_blueprint_library()

        # vehicles
        self.vehicle_bps = self.world.get_blueprint_library().filter('*vehicle*')
        self.vehicle_bps = [bp for bp in self.vehicle_bps if bp.get_attribute('number_of_wheels').as_int() > 2]
        
        self.cav_bp = self.world.get_blueprint_library().find("vehicle.mercedes.coupe")
        # bev rgb_camera
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute("image_size_x", str(512))
        self.camera_bp.set_attribute("image_size_y", str(512))
        self.camera_bp.set_attribute("fov", str(135))

        # surround camera bp
        self.svc_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # The FOV value is from opv2v
        self.svc_bp.set_attribute('fov', str(110))

        # depth camera bp
        self.svc_depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        self.svc_depth_bp.set_attribute('fov', str(110))

        # sem camera bp
        self.svc_sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.svc_sem_bp.set_attribute('fov', str(110))

        # fisheye camera bp
        self.fisheye_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.fisheye_bp.set_attribute("image_size_x", str(1024))
        self.fisheye_bp.set_attribute("image_size_y", str(1024))
        self.fisheye_bp.set_attribute("fov", str(90))
    
        # semantic camera
        self.semantic_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')

        save_path = join(save_path, env_name)
        self.save_path = save_path
        self.vehicle_save_path = {}

        # BEV camera intrinsic
        self.rgb_intrinsic, self.img_x, self.img_y = get_intrinsic_matrix(self.camera_bp)

        # Depth Camera intrinsic
        self.depth_intrinsic, _, _ = get_intrinsic_matrix(self.svc_depth_bp)
        
        # SVC camera intrinsic
        self.svc_intrinsic, _, _ = get_intrinsic_matrix(self.svc_bp)

        self.bev_shape = bev_shape
        # use fisheye
        self.use_fisheye = use_fisheye

        self.nv_trans = neighbor_trans
        self.cav_trans = cav_trans
        self.use_autopilot = use_autopilot

        if self.bev_cam_trans == []:
            self.spawn_agents()
        else:
            self.spawn_agents_in_specific_pos()
        

        self.frame_idx = []
        self.absence_frame = set()

    def spawn_agents_in_specific_pos(self):
        surround_camera_trans = create_surrond_camera_trans()

        num = 0
        for trans in self.cav_trans:
            vehicle = self.world.try_spawn_actor(self.cav_bp, trans)
            if self.use_autopilot[num]:
                vehicle.set_autopilot(True)

            vehicle_id = vehicle.id
            
            self.cav_ids.append(vehicle_id)
            self.vehicles[vehicle_id] = vehicle

            # We spawn the camera and attach it to our ego vehicle
            sem_camera = self.world.spawn_actor(self.semantic_camera_bp, self.bev_trans, attach_to=vehicle)
            rgb_camera = self.world.spawn_actor(self.camera_bp, self.bev_trans, attach_to=vehicle)

            img_callback_with_params = lambda image, vehicle_id = vehicle_id: self.bev_rgb_camera_callback(image, vehicle_id)
            sem_callback_with_params = lambda image, vehicle_id = vehicle_id : self.semantic_camera_callback(image, vehicle_id)

            rgb_camera.listen(img_callback_with_params)
            sem_camera.listen(sem_callback_with_params)

            self.semantic_cameras[vehicle_id] = sem_camera
            self.rgb_cameras[vehicle_id] = rgb_camera
            self.svc_cameras[vehicle_id] = {}
            self.svc_depth_cameras[vehicle_id] = {}
            self.svc_sem_cameras[vehicle_id] = {}
            self.svc_depth[vehicle_id] = {}
            self.svc_vehicle_mask[vehicle_id] = {}
 
            for cam_id, trans in surround_camera_trans.items():
                svc_camera = self.world.spawn_actor(self.svc_bp, trans, attach_to=vehicle)
                svc_callback_with_params = lambda image, ego_vehicle_id = vehicle_id, cam_id = cam_id : self.svc_callback(image, ego_vehicle_id, cam_id)
                svc_camera.listen(svc_callback_with_params)
                self.svc_cameras[vehicle_id][cam_id] = svc_camera

                svc_depth_camera = self.world.spawn_actor(self.svc_depth_bp, trans, attach_to=vehicle)
                svc_depth_callback_with_params = lambda image, ego_vehicle_id = vehicle_id, cam_id = cam_id : self.svc_depth_camera_callback(image, ego_vehicle_id, cam_id)
                svc_depth_camera.listen(svc_depth_callback_with_params)
                self.svc_depth_cameras[vehicle_id][cam_id] = svc_depth_camera

                svc_sem_camera = self.world.spawn_actor(self.svc_sem_bp, trans, attach_to=vehicle)
                svc_sem_callback_with_params = lambda image, ego_vehicle_id = vehicle_id, cam_id = cam_id : self.svc_semantic_camera_callback(image, ego_vehicle_id, cam_id)
                svc_sem_camera.listen(svc_sem_callback_with_params)
                self.svc_sem_cameras[vehicle_id][cam_id] = svc_sem_camera

                if self.use_fisheye:
                    self.fisheye_cameras[vehicle_id] = {}
                    cams = {}
                    fisheye_cameras = create_fisheye_camera_trans(trans.location, trans.rotation)
                    for direction in fisheye_cameras:
                        fisheye = self.world.spawn_actor(self.fisheye_bp,
                                                            fisheye_cameras[direction] 
                                                            , attach_to=vehicle)
                        fisheye_callback_with_params = lambda image, ego_vehicle_id = vehicle_id, direction = direction\
                              : self.fisheye_callback(image, ego_vehicle_id, cam_id, direction)
                        fisheye.listen(fisheye_callback_with_params)

                        cams[direction] = fisheye
                    self.fisheye_cameras[vehicle_id][cam_id] = cams

            self.vehicle_save_path[vehicle_id] = join(self.save_path, f"{vehicle_id}")
            if not os.path.exists(self.vehicle_save_path[vehicle_id]):
                os.makedirs(self.vehicle_save_path[vehicle_id])
            num += 1

        for trans in self.nv_trans:
            vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_bps), trans)
            
            if vehicle is None:
                continue

            if self.use_autopilot[num]:
                vehicle.set_autopilot(True)
           
            vehicle_id = vehicle.id

            self.vehicles[vehicle_id] = vehicle
            num += 1

    def spawn_agents(self):
        """
            This function spawn agents in the map without given speficic spawn points     
        """
        
        surround_camera_trans = create_surrond_camera_trans()
        num = 0
        while num < self.cav_num:
            sp = random.choice(self.spawn_points)

            vehicle = self.world.try_spawn_actor(self.cav_bp, sp)
            if vehicle == None:
                continue
            self.spawn_points.remove(sp)
            vehicle.set_autopilot(True)
            vehicle_id = vehicle.id
            
            self.cav_ids.append(vehicle_id)
            self.vehicles[vehicle_id] = vehicle

            # We spawn the camera and attach it to our ego vehicle
            sem_camera = self.world.spawn_actor(self.semantic_camera_bp, self.bev_trans, attach_to=vehicle)
            rgb_camera = self.world.spawn_actor(self.camera_bp, self.bev_trans, attach_to=vehicle)

            img_callback_with_params = lambda image, vehicle_id = vehicle_id: self.bev_rgb_camera_callback(image, vehicle_id)
            sem_callback_with_params = lambda image, vehicle_id = vehicle_id : self.semantic_camera_callback(image, vehicle_id)

            rgb_camera.listen(img_callback_with_params)
            sem_camera.listen(sem_callback_with_params)

            self.semantic_cameras[vehicle_id] = sem_camera
            self.rgb_cameras[vehicle_id] = rgb_camera
            self.svc_cameras[vehicle_id] = {}
            self.svc_depth_cameras[vehicle_id] = {}
            self.svc_sem_cameras[vehicle_id] = {}
            self.svc_depth[vehicle_id] = {}
            self.svc_vehicle_mask[vehicle_id] = {}
 
            for cam_id, trans in surround_camera_trans.items():
                svc_camera = self.world.spawn_actor(self.svc_bp, trans, attach_to=vehicle)
                svc_callback_with_params = lambda image, ego_vehicle_id = vehicle_id, cam_id = cam_id : self.svc_callback(image, ego_vehicle_id, cam_id)
                svc_camera.listen(svc_callback_with_params)
                self.svc_cameras[vehicle_id][cam_id] = svc_camera

                svc_depth_camera = self.world.spawn_actor(self.svc_depth_bp, trans, attach_to=vehicle)
                svc_depth_callback_with_params = lambda image, ego_vehicle_id = vehicle_id, cam_id = cam_id : self.svc_depth_camera_callback(image, ego_vehicle_id, cam_id)
                svc_depth_camera.listen(svc_depth_callback_with_params)
                self.svc_depth_cameras[vehicle_id][cam_id] = svc_depth_camera

                svc_sem_camera = self.world.spawn_actor(self.svc_sem_bp, trans, attach_to=vehicle)
                svc_sem_callback_with_params = lambda image, ego_vehicle_id = vehicle_id, cam_id = cam_id : self.svc_semantic_camera_callback(image, ego_vehicle_id, cam_id)
                svc_sem_camera.listen(svc_sem_callback_with_params)
                self.svc_sem_cameras[vehicle_id][cam_id] = svc_sem_camera

                if self.use_fisheye:
                    self.fisheye_cameras[vehicle_id] = {}
                    cams = {}
                    fisheye_cameras = create_fisheye_camera_trans(trans.location, trans.rotation)
                    for direction in fisheye_cameras:
                        fisheye = self.world.spawn_actor(self.fisheye_bp,
                                                            fisheye_cameras[direction] 
                                                            , attach_to=vehicle)
                        fisheye_callback_with_params = lambda image, ego_vehicle_id = vehicle_id, direction = direction\
                              : self.fisheye_callback(image, ego_vehicle_id, cam_id, direction)
                        fisheye.listen(fisheye_callback_with_params)

                        cams[direction] = fisheye
                    self.fisheye_cameras[vehicle_id][cam_id] = cams

            self.vehicle_save_path[vehicle_id] = join(self.save_path, f"{vehicle_id}")
            if not os.path.exists(self.vehicle_save_path[vehicle_id]):
                os.makedirs(self.vehicle_save_path[vehicle_id])
            num += 1

        num = 0
        while num < self.npc_num:
            sp = random.choice(self.spawn_points)
            vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_bps), sp)
            if vehicle == None:
                print("SPawn fail")
                continue
            self.spawn_points.remove(sp)
            vehicle.set_autopilot(True)
            vehicle_id = vehicle.id

            self.vehicles[vehicle_id] = vehicle
            num += 1

    def bev_rgb_camera_callback(self, image, ego_vehicle_id):
        data = np.frombuffer(image.raw_data, dtype=np.uint8)
        data = np.reshape(data, (image.height, image.width, 4))
        camera_trans = image.transform
        self.bev_cam_trans[ego_vehicle_id] =  camera_trans
        cv2.imwrite(join(self.vehicle_save_path[ego_vehicle_id] ,"{}_bev_rgb.png".format(image.frame)), deepcopy(data))
        
        bev_dynamic = np.zeros((image.height, image.width), dtype=np.uint8)
        for vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            vehicle_trans = vehicle.get_transform()

            bbox = bounding_box(vehicle.bounding_box.extent, vehicle_trans)
            bbox = turn_world_cood_to_camera_coord(bbox, camera_trans, self.rgb_intrinsic)
            
            bbox = np.clip(bbox, np.array([0,0]), np.array([self.img_y, self.img_x])).astype(np.int32)
            if (bbox[0][0]- bbox[2][0]) * (bbox[0][1] - bbox[2][1]) == 0:
                continue
            cv2.rectangle(bev_dynamic, bbox[0], bbox[2], 255, thickness=cv2.FILLED)
        
        bev_dynamic = cv2.resize(bev_dynamic,  dsize=self.bev_shape)

        cv2.imwrite(join(self.vehicle_save_path[ego_vehicle_id] ,"{}_bev_dynamic.png".format(image.frame)), bev_dynamic)
        cv2.imwrite(join(self.vehicle_save_path[ego_vehicle_id] ,"{}_bev_visibility_corp.png".format(image.frame)), bev_dynamic)


    # 定义语义分割相机回调函数
    def semantic_camera_callback(self, data, ego_vehicle_id):
        # 获取语义分割图像数据
        semantic_data = np.frombuffer(data.raw_data, dtype = np.uint8)
        semantic_data = np.reshape(semantic_data, (data.height, data.width, 4))[..., 2]
        
        road_line = (np.where(semantic_data == 6, 1, 0) * 255).astype(np.uint8)
        road_line = cv2.resize(road_line, dsize=self.bev_shape)
        cv2.imwrite(join(self.vehicle_save_path[ego_vehicle_id], "{}_bev_lane.png".format( data.frame)), road_line)

        road = (np.bitwise_or(np.bitwise_or(np.where(semantic_data == 7, 1, 0), 
                             np.where(semantic_data == 6, 1, 0)),\
                             np.where(semantic_data == 10, 1, 0)) * 255).astype(np.uint8)
        road = cv2.resize(road, dsize=self.bev_shape)
        cv2.imwrite(join(self.vehicle_save_path[ego_vehicle_id], "{}_bev_static.png".format( data.frame)), road)

        data.convert(carla.ColorConverter.CityScapesPalette)
        data.save_to_disk(join(self.vehicle_save_path[ego_vehicle_id], "{}_bev_seg_dynamic.png".format(data.frame)))

    def svc_semantic_camera_callback(self, data, ego_vehicle_id, cam_id):
        # 获取语义分割图像数据
        semantic_data = np.frombuffer(data.raw_data, dtype = np.uint8)
        semantic_data = np.reshape(semantic_data, (data.height, data.width, 4))[..., 2]
        sem_mask =  np.where(semantic_data == 10, 1, 0).astype(np.uint8)

        output = cv2.connectedComponents(sem_mask, connectivity=8, ltype=cv2.CV_32S)#计算连同域

        num_labels = output[0]
        labels = output[1]

        for label in range(1, num_labels):
            mask = np.where(labels == label)
            if len(mask[0]) < 100:
                labels[mask] = 0

        # cv2.imwrite(join(self.vehicle_save_path[ego_vehicle_id], f"{data.frame}_{cam_id}_con.png"), (labels* 10).astype(np.uint8)) 

        self.svc_vehicle_mask[ego_vehicle_id][cam_id] = {
            label : np.where(labels == label) for label in range(1, num_labels) if len(np.where(labels == label)[0]) > 100
        }

    def svc_depth_camera_callback(self, data, ego_vehicle_id, cam_id):
        raw_depth = np.frombuffer(data.raw_data, dtype = np.uint8).reshape((data.height, data.width, 4)).astype(float)
        normalized = (raw_depth[..., 2] + raw_depth[..., 1] * 256 + raw_depth[..., 0] * 256 * 256) / (256 * 256 * 256 - 1)
    
        in_meters = 1000 * normalized
        trans = data.transform
        self.svc_depth[ego_vehicle_id][cam_id] = (in_meters, trans)

        # data.save_to_disk(join(self.vehicle_save_path[ego_vehicle_id], f"{data.frame}_{cam_id}_depth.png"),carla.ColorConverter.Depth)


    def svc_callback(self, image, ego_vehicle_id, cam_id):
        image.save_to_disk(join(self.vehicle_save_path[ego_vehicle_id], "{}_camera{}.png".\
                                format(image.frame, cam_id)))
        
    def fisheye_callback(self, image, ego_vehicle_id, cam_id, direction):
        image.save_to_disk(join(self.vehicle_save_path[ego_vehicle_id], "{}_camera{}_{}.png".\
                                format(image.frame, cam_id, direction)))

    def simulate(self, time_steps):
        for i in range(time_steps):
            print("step {}".format(i))
            self.world.tick()  # 推进仿真的一个时间步长
            frame_idx = self.world.get_snapshot().frame
            self.frame_idx.append(frame_idx)
            self.record_ego_to_yaml(frame_idx)
            time.sleep(1)  # 可选：降低循环速率，避免过度占用CPU

            self.validation_check(frame_idx)
            self.sample_neighbor_vehicle(frame_idx)

        self.remove_absence_frame()

    def record_ego_to_yaml(self, frame):
        for ego_vehicle_id in self.cav_ids:
            """
                Set every vehicle as ego vheicle one by one
            """
            ego_record = self.record_ego_vehicle(ego_vehicle_id)
            with open(join(self.vehicle_save_path[ego_vehicle_id], "{}.yaml".format(frame)), 'w') as outfile:
                yaml.dump(ego_record, outfile)
            
    def record_ego_vehicle(self, ego_vehicle_id):
        ego_record = {}
        
        # reocrd svc transform
        svcs = self.svc_cameras[ego_vehicle_id]
        
        for cam_id in svcs:
            cam_geo = {}
            cam = svcs[cam_id]
            
            cam_trans = cam.get_transform()
            extrinsic = cam_trans.get_matrix()
            intrinsic = self.svc_intrinsic.tolist()

            cam_geo["cords"] = turn_trans_to_list(cam_trans)
            cam_geo["extrinsic"] = extrinsic
            cam_geo["intrinsic"] = intrinsic
            ego_record["camera{}".format(cam_id)] = cam_geo

        # record ego_vehicle
        ego_vehicle = self.vehicles[ego_vehicle_id]
        ego_record["ego_speed"] = turn_velocity_to_speed(ego_vehicle.get_velocity())
        ego_record["true_ego_pos"] = turn_trans_to_list(ego_vehicle.get_transform())

        ego_record["lidar_pose"] = [0] * 6
        ego_record["vehicles"] = {}

        # record other vheicles
        for vehicle_id in self.vehicles:
            if vehicle_id == ego_vehicle_id:
                continue
            ego_record["vehicles"][vehicle_id] = \
                    self.single_vehicle_state(self.vehicles[vehicle_id])
        return ego_record
        
    def single_vehicle_state(self, vehicle):
        state = {}
        vehicle_trans = turn_trans_to_list(vehicle.get_transform())

        state["angle"] = vehicle_trans[3:]
        state["speed"] = turn_velocity_to_speed(vehicle.get_velocity())
        state["location"] = vehicle_trans[:3]
        state["extent"] = turn_extent_to_list(vehicle.bounding_box.extent)
        state["center"] = turn_extent_to_list(vehicle.bounding_box.location)
        return state

    def destroy_world(self):
        for i, camera in self.semantic_cameras.items():
            camera.stop()

        for i, camera in self.rgb_cameras.items():
            camera.stop()

        for i in self.fisheye_cameras:
            cams = self.fisheye_cameras[i]
            for cam_id in cams:
                for direction in cams[cam_id]:
                    cams[cam_id][direction].stop()

        for vehicle_id , cams in self.svc_cameras.items():
            for cam_id , cam in cams.items():
                cam.stop()

        # 摧毁actors
        for actor in self.world.get_actors():
            actor.destroy()

        # vehicles
        self.vehicles = {}

        # rgb cameras
        self.rgb_cameras = {}

        # semantic cameras
        self.semantic_cameras = {}

    def sample_neighbor_vehicle(self, frame):
        # store vehicle coodinations
        vehicle_location = []
        for vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            vehicle_trans = vehicle.get_transform()
            vehicle_location.append([vehicle_trans.location.x, vehicle_trans.location.y])

        vehicle_ids = np.array(list(self.vehicles.keys()))
        vehicle_location = np.array(vehicle_location).reshape((-1, 1, 2))

        for ego_vehicle_id in self.cav_ids:
            
            svc_depth, svc_vehicle_mask = self.svc_depth[ego_vehicle_id], self.svc_vehicle_mask[ego_vehicle_id]
            center_points = []

            for cam_id in svc_depth:
                masks = svc_vehicle_mask[cam_id]
                depth, trans = svc_depth[cam_id]
                # depth_map = np.zeros_like(depth)

                # for _, mask in masks.items():
                #     depth_map[mask] = depth[mask]
                
                # cv2.imwrite(join(self.vehicle_save_path[ego_vehicle_id], "{}_{}_dep.png".format(frame, cam_id)), ((depth_map / 1000)*255).astype(np.uint8))
                if len(masks) == 0:
                    continue

                depth = np.array(depth[list(masks.values())[0]]) if len(masks) == 1 else\
                      np.concatenate([depth[mask] for _, mask in masks.items()])
                
                mask_pixel = np.array(list(masks.values())[0]) if len(masks) == 1 else\
                      np.concatenate([np.array(mask) for _, mask in masks.items()], axis = 1)
                mask_pixel = (mask_pixel.T[depth < 800]).T
                depth = depth[depth < 800]

                if len(depth) == 0:
                    continue

                # print(np.min(depth), np.max(depth))
                world_cood = turn_camera_cood_to_world_coord(mask_pixel, 
                                                             trans, 
                                                             self.depth_intrinsic, 
                                                             depth)
                i = 0
                for label in masks:
                    center_points.append(np.mean(world_cood[i:i+len(masks[label])], axis =0))
                    i += len(masks[label])

            vehicle_trans = self.vehicles[ego_vehicle_id].get_transform()
            center_points.append(np.array([vehicle_trans.location.x, vehicle_trans.location.y]))
            center_points = np.array(center_points).reshape((1, -1, 2))
            
            dists = np.sqrt(np.sum((center_points - vehicle_location)**2, axis = 2))
            nv_idx = np.where(np.sum((dists < 10), axis = 1) > 0)
            nv_ids = vehicle_ids[nv_idx]
            
            camera_trans = self.bev_cam_trans[ego_vehicle_id]
            # cps = turn_world_cood_to_camera_coord(np.concatenate([center_points[0], np.zeros((len(center_points[0]), 1)), np.ones((len(center_points[0]),1))], axis = 1).T,
            #                                                         camera_trans, self.rgb_intrinsic)
            
            bev_dynamic = np.zeros((512, 512), dtype=np.uint8)
            for vehicle_id in nv_ids:
                vehicle = self.vehicles[vehicle_id]
                vehicle_trans = vehicle.get_transform()

                bbox = bounding_box(vehicle.bounding_box.extent, vehicle_trans)
                bbox = turn_world_cood_to_camera_coord(bbox, camera_trans, self.rgb_intrinsic)
                
                bbox = np.clip(bbox, np.array([0,0]), np.array([self.img_y, self.img_x])).astype(np.int32)
                if (bbox[0][0]- bbox[2][0]) * (bbox[0][1] - bbox[2][1]) == 0:
                    continue
                cv2.rectangle(bev_dynamic, bbox[0], bbox[2], 255, thickness=cv2.FILLED)
            
            # for cp in cps:
            #     cv2.circle(bev_dynamic, (cp[0],cp[1]), 5, 255, -1)
            bev_dynamic = cv2.resize(bev_dynamic,  dsize=self.bev_shape)
            cv2.imwrite(join(self.vehicle_save_path[ego_vehicle_id] ,"{}_bev_visibility.png".format(frame)), bev_dynamic)

    def validation_check(self, frame_id):
        checklist = ["_bev_dynamic.png", "_bev_lane.png", "_bev_static.png",
                     "_bev_visibility_corp.png", "_bev_rgb.png",
                     ".yaml", "_bev_seg_dynamic.png",
                     "_camera0.png", "_camera1.png", "_camera2.png", "_camera3.png"]

        for vehicle_id in self.cav_ids:
            for file_suffix in checklist:
                if not os.path.exists(join(self.save_path, 
                                        f"{vehicle_id}", f"{frame_id}"+file_suffix)):
                    print("{} didn`t exist".format(join(self.save_path, 
                                        f"{vehicle_id}", f"{frame_id}"+file_suffix)))
                    self.absence_frame.add(frame_id)
                    continue

                if file_suffix.endswith("png"):
                    img = cv2.imread(join(self.save_path, 
                                        f"{vehicle_id}", f"{frame_id}"+file_suffix))
                    try:
                        cv2.resize(img, (2,2))
                    except:
                        print("{} is broken".format(join(self.save_path, 
                                        f"{vehicle_id}", f"{frame_id}"+file_suffix)))
                        if frame_id - 1 in self.frame_idx:
                            cv2.imwrite(
                                join(self.save_path, 
                                        f"{vehicle_id}", f"{frame_id}"+file_suffix),
                                cv2.imread(join(self.save_path, 
                                        f"{vehicle_id}", f"{frame_id-1}"+file_suffix))
                            )


    def remove_absence_frame(self):
        checklist = ["_bev_dynamic.png", "_bev_lane.png", "_bev_static.png",
                     "_bev_visibility_corp.png", "_bev_visibility.png", "_bev_rgb.png",
                     ".yaml", "_bev_seg_dynamic.png",
                     "_camera0.png", "_camera1.png", "_camera2.png", "_camera3.png"]
        for frame_id in self.absence_frame:
            for vehicle_id in self.cav_ids:
                for file_suffix in checklist:
                        os.remove(join(self.save_path, 
                                            f"{vehicle_id}", f"{frame_id}"+file_suffix))

