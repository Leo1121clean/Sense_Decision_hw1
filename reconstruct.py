import numpy as np
import open3d as o3d
import cv2
import math
from sklearn.neighbors import NearestNeighbors

final_count = 1 #counting point_cloud reconstruction
img2pcd_count = 1 #counting image_to_point_cloud
voxel_size = 0.05  # means 5cm for this dataset

#####depth_image_to_point_cloud#####
save_rgb = []
save_pixel = []
def get_pointcloud(img2pcd_count):
    
    save_rgb.clear()
    save_pixel.clear()
    
    img_rgb = cv2.imread('img/rgb/rgb' + str(img2pcd_count) + '.png')
    img_depth = cv2.imread('img/depth/depth' + str(img2pcd_count) + '.png')
    
    fov = 90
    f = float(512/2*(1/math.tan(fov/180*math.pi/2)))
    
    for i in range(0,512):
        for j in range(0,512):
            x = (img_depth[i,j][0]/25.5)*(j-256)/f
            y = (img_depth[i,j][0]/25.5)*(i-256)/f
            
            if y>(-0.6):  #去除天花板
                save_pixel.append([x,y,img_depth[i,j][0]/25.5])
                save_rgb.append([img_rgb[i][j][2]/255,img_rgb[i][j][1]/255,img_rgb[i][j][0]/255])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(save_pixel)
    pcd.colors = o3d.utility.Vector3dVector(save_rgb)
    o3d.io.write_point_cloud('pcd/' + str(img2pcd_count) +  '.pcd',pcd)
    print("Save Point Cloud " + str(img2pcd_count))


#####初始化點雲#####
def prepare_dataset(voxel_size,source,target):
    
    print(":: Load two point clouds and disturb initial pose. " + str(final_count))

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # source.estimate_normals()
    # target.estimate_normals()
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


#####點雲前處理#####
def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    #加上法向量(後面point to plane ICP會用到)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


#####global registration#####
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


#####Local refinement, 這裡做point-to-plane ICP#####
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

#####將轉移後的source存到下一次使用的target#####
def new_target(source_down,T):
    return source_down.transform(T)


#####整合過的獨立ICP function#####
def local_icp(voxel_size, total_count):
    
    global final_count
    final = []
    
    estimate_point = [[0,0,0]]
    estimate_color = []
    origin = np.array([[0],[0],[0],[1]])

    while True:
        
        source = o3d.io.read_point_cloud('pcd/' + str(final_count+1) + '.pcd')
        if final_count == 1:
            target = o3d.io.read_point_cloud('pcd/1.pcd')
        
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,source,target)
        result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
        result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size, result_ransac)
        T = result_icp.transformation
        
        # 此部份為自己寫的ICP,目前還有錯誤未解決
        # source_array = np.asarray(source_down.points)
        # target_array = np.asarray(target_down.points)
        # T = icp(source_array, target_array, result_ransac.transformation, times=20, threshold=0.001)
        
        if final_count == 1:
            final.append(target_down)
            
        target = new_target(source_down,T)
        final.append(source_down)
        
        #make estimated trajectory
        origin_trans = np.dot(T,origin)
        origin_trans = np.transpose(origin_trans).tolist()
        estimate_point.append([origin_trans[0][0], origin_trans[0][1], origin_trans[0][2]])
        estimate_color.append([255,0,0])
        estimate_lines = [[i,i+1] for i in range(total_count-1)]
        
        if final_count == total_count-1:
            break
        
        final_count = final_count + 1
    
    return final, estimate_point, estimate_color, estimate_lines


#####ground_truth_trajectory#####
def ground_truth_trajectory(total_count):
    
    gt_count = 0
    x_list = []
    y_list = []
    z_list = []
    gt_point = []
    gt_color = []
    with open('ground_xyz.txt', 'r') as infile:
        for line in infile.readlines():
            s = line.split(' ')
            x = s[0:1]
            y = s[1:2]
            z = s[2:3]
            x_list.append(float(x[0]))
            y_list.append(float(y[0]))
            z_list.append(float(z[0]))
    infile.close()
    
    while gt_count < total_count:
        
        a = x_list[0]
        b = y_list[0]
        c = z_list[0]
        for i in range(len(x_list)):
            gt_point.append([x_list[i]-a, y_list[i]-b, -(z_list[i]-c)])
        
        gt_color.append([0,0,0])
        gt_lines = [[i,i+1] for i in range(total_count-1)]
        
        gt_count = gt_count + 1

    return gt_point, gt_color, gt_lines

#####distance between ground truth trajectory and estimated trajectory####
def calculate_distance(total_count):
    cal_count = 0
    dis = 0
    while cal_count < total_count:
        x = gt_point[cal_count][0] - estimate_point[cal_count][0]
        y = gt_point[cal_count][1] - estimate_point[cal_count][1]
        z = gt_point[cal_count][2] - estimate_point[cal_count][2]
        dis = dis + math.sqrt(x**2+y**2+z**2)
        cal_count = cal_count +1
    
    mean_dis = dis/total_count
    print("Mean distance: " + str(mean_dis))


##########################ICP algorithm(still has error)##########################
#find T with SVD
def best_fit_transform(source, target):
    
    dimension = 3

    # 將所有點扣掉質心
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_new = source - source_center
    target_new = target - target_center

    # rotation matrix
    W = np.dot(source_new.T, target_new)
    U, S, VT = np.linalg.svd(W)
    R = np.dot(VT.T, U.T)

    # add translation and rotation into T
    t = target_new.T - np.dot(R,source_new.T)
    T = np.identity(dimension+1)
    T[:dimension, :dimension] = R
    T[:dimension, dimension] = t

    return T, R, t

#尋找鄰近點(source to target)
def nearest_neighbor(source, target):

    neighbor = NearestNeighbors(n_neighbors=1)
    neighbor.fit(target)
    distance, find = neighbor.kneighbors(source, return_distance=True)
    return distance.ravel(), find.ravel()

#icp algorithm function
def icp(source, target, ransac_T, times, threshold):

    dimension = 3

    source_new = np.ones((dimension+1,source.shape[0]))
    target_new = np.ones((dimension+1,target.shape[0]))
    source_new[:dimension,:] = np.copy(source.T)
    target_new[:dimension,:] = np.copy(target.T)

    source_new = np.dot(ransac_T, source_new)

    old_error = 0
    for i in range(times):
        #尋找source和target的鄰近點
        distance, find = nearest_neighbor(source_new[:dimension,:].T, target_new[:dimension,:].T)

        # 求此迭代產生的T
        T,_,_ = best_fit_transform(source_new[:dimension,:].T, target_new[:dimension,find].T)

        # 更新source
        source_new = np.dot(T, source_new)

        # 誤差判斷，小於閥值就停止尋找
        mean_error = np.mean(distance)
        if np.abs(old_error - mean_error) < threshold:
            break
        old_error = mean_error

    # 計算經迭代後產生的T
    T,_,_ = best_fit_transform(source, source_new[:dimension,:].T)

    return T
##########################ICP algorithm(still has error)##########################


if __name__ == "__main__":
    
    #讀取圖片張數
    with open('count.txt', 'r') as infile:
        total_count = int(infile.read())
    
    #生成點雲
    while img2pcd_count <= total_count:
        get_pointcloud(img2pcd_count)
        img2pcd_count = img2pcd_count+1    
    
    #執行ICP重建函式(包含estimated trajectory)
    final, estimate_point, estimate_color, estimate_lines = local_icp(voxel_size, total_count)   
    
    #建立estimated trajectory
    estimate = o3d.geometry.LineSet()
    estimate.lines = o3d.utility.Vector2iVector(estimate_lines)
    estimate.colors = o3d.utility.Vector3dVector(estimate_color)
    estimate.points = o3d.utility.Vector3dVector(estimate_point)
    
    #建立ground truth trajectory
    gt_point, gt_color, gt_lines = ground_truth_trajectory(total_count)
    ground = o3d.geometry.LineSet()
    ground.lines = o3d.utility.Vector2iVector(gt_lines)
    ground.colors = o3d.utility.Vector3dVector(gt_color)
    ground.points = o3d.utility.Vector3dVector(gt_point)

    #計算兩種路徑之平均誤差
    calculate_distance(total_count)

    #建立3D點雲資料
    final_pcd = o3d.geometry.PointCloud()
    for i in range(len(final)):
        final_pcd = final_pcd + final[i]
    
    #視覺化所有點雲資料及存檔
    o3d.visualization.draw_geometries([final_pcd, estimate, ground])
    o3d.io.write_point_cloud('final.pcd',final_pcd)
