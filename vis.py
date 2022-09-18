# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
from __future__ import division
import os
import numpy as np
import cv2
 
def project_velo2rgb(velo,calib):
    T=np.zeros([4,4],dtype=np.float32)
    T[:3,:]=calib['Tr_velo2cam']
    T[3,3]=1
    R=np.zeros([4,4],dtype=np.float32)
    R[:3,:3]=calib['R0']
    R[3,3]=1
    num=len(velo)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for i in range(len(velo)):
        box3d=np.ones([8,4],dtype=np.float32)
        box3d[:,:3]=velo[i]
        M=np.dot(calib['P2'],R)
        M=np.dot(M,T)
        box2d=np.dot(M,box3d.T)
        box2d=box2d[:2,:].T/box2d[2,:].reshape(8,1)
        projections[i] = box2d
    return projections
 
def draw_rgb_projections(image, projections, color=(255,255,255), thickness=2, darker=1):
 
    img = image.copy()*darker
    num=len(projections)
    forward_color=(255,255,0)
    for n in range(num):
        qs = projections[n]
        for k in range(0,4):
            i,j=k,(k+1)%4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
            i,j=k+4,(k+1)%4 + 4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
            i,j=k,k+4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[3,0],qs[3,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[7,0],qs[7,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[6,0],qs[6,1]), (qs[2,0],qs[2,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[3,0],qs[3,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[3,0],qs[3,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)
    return img
 
#过滤指定范围之外的点和目标框
def get_filtered_lidar(lidar, boxes3d=None):
    xrange = (0, 70.4)
    yrange = (-40, 40)
    zrange = (-3, 1)
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    filter_x = np.where((pxs >= xrange[0]) & (pxs < xrange[1]))[0]
    filter_y = np.where((pys >= yrange[0]) & (pys < yrange[1]))[0]
    filter_z = np.where((pzs >= zrange[0]) & (pzs < zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)
    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= xrange[0]) & (boxes3d[:, :, 0] < xrange[1])
        box_y = (boxes3d[:, :, 1] >= yrange[0]) & (boxes3d[:, :, 1] < yrange[1])
        box_z = (boxes3d[:, :, 2] >= zrange[0]) & (boxes3d[:, :, 2] < zrange[1])
        box_xyz = np.sum(box_x & box_y & box_z,axis=1)
        return lidar[filter_xyz], boxes3d[box_xyz>0]
    return lidar[filter_xyz]
 
def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)
    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)
    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}
 
def box3d_cam_to_velo(box3d, Tr):
    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)
 
    def ry_to_rz(ry):
        angle = -ry - np.pi / 2
        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle
        return angle
 
    h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)
    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])
    rz = ry_to_rz(ry)
    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])
    velo_box = np.dot(rotMat, Box)
    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = cornerPosInVelo.transpose()
    return box3d_corner.astype(np.float32)
 
def load_kitti_label(label_file, Tr):
    with open(label_file,'r') as f:
        lines = f.readlines()
    gt_boxes3d_corner = []
    num_obj = len(lines)
    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        if obj_class not in ['Car']:
            continue
        box3d_corner = box3d_cam_to_velo(obj[8:], Tr)
        gt_boxes3d_corner.append(box3d_corner)
    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1,8,3)
    return gt_boxes3d_corner
 
def test():
    lidar_path = os.path.join('./data/KITTI/training', "velodyne/")
    calib_path = os.path.join('./data/KITTI/training', "calib/")
    label_path = os.path.join('./data/KITTI/training', "label_2/")
    image_path = os.path.join('./data/KITTI/training', "image_2/")

    for i in range(0,11):
        s = str(i)
        t = '0'*(6-len(s))+s
        print(t)
        lidar_file = lidar_path + '/' + t + '.bin'
        calib_file = calib_path + '/' + t + '.txt'
        label_file = label_path + '/' + t + '.txt'
        image_file = image_path + '/' + t + '.png'
    
        image = cv2.imread(image_file)
        #加载雷达数据
        print("Processing: ", lidar_file)
        lidar = np.fromfile(lidar_file, dtype=np.float32)
        lidar = lidar.reshape((-1, 4))
    
        #加载标注文件
        calib = load_kitti_calib(calib_file)
        #标注转三维目标检测框
        gt_box3d = load_kitti_label(label_file, calib['Tr_velo2cam'])
    
        #过滤指定范围之外的点和目标框
        lidar, gt_box3d = get_filtered_lidar(lidar, gt_box3d)
    
        # view in point cloud，可视化
        gt_3dTo2D = project_velo2rgb(gt_box3d, calib)
        img_with_box = draw_rgb_projections(image,gt_3dTo2D, color=(0,0,255),thickness=1)
        w = 'box'+t+'.png'
        print(w)
        cv2.imwrite(w, img_with_box)
        #cv2.imshow(w, img_with_box)
        #cv2.waitKey(0)
 
if __name__ == '__main__':
    test()