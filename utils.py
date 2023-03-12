import numpy as np
import cv2

def load_images():
    img = []
    img1 = cv2.imread('Media/image_1.jpg')
    img1 = cv2.resize(img1, (int(3024/5), int(4032/5)))
    img.append(img1)
    img2 = cv2.imread('Media/image_2.jpg')
    img2 = cv2.resize(img2, (int(3024/5), int(4032/5)))
    img.append(img2)
    img3 = cv2.imread('Media/image_3.jpg')
    img3 = cv2.resize(img3, (int(3024/5), int(4032/5)))
    img.append(img3)
    return img

def good_points_matches(img1 , img2 , is_destination_right_image):           
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    
    if is_destination_right_image:
        matches = bf.knnMatch(descriptors_1 ,descriptors_2, k=2)
        # Lowe's ratio test
        good_points = []
        for m,n in matches:
            if m.distance < 0.4*n.distance:
                good_points.append(m)
        if len(good_points)>10:
            source_points = np.float32([keypoints_1[m.queryIdx].pt for m in good_points ]).reshape(-1,1,2)
            destination_points = np.float32([keypoints_2[m.trainIdx].pt for m in good_points ]).reshape(-1,1,2)  
    else :
        matches = bf.knnMatch(descriptors_2 ,descriptors_1, k=2)
        # Lowe's ratio test
        good_points = []
        for m,n in matches:
            if m.distance < 0.4*n.distance:
                good_points.append(m)
        if len(good_points)>10:
            source_points = np.float32([ keypoints_2[m.queryIdx].pt for m in good_points ]).reshape(-1,1,2)
            destination_points = np.float32([ keypoints_1[m.trainIdx].pt for m in good_points ]).reshape(-1,1,2) 
            
    return source_points , destination_points , len(good_points)

def estimate_homography(source, destination):
    source_homography = np.hstack((source, np.ones((source.shape[0], 1))))
    A = np.array([np.block([[source_homography[n], np.zeros(3), -destination[n, 0] * source_homography[n]],
                            [np.zeros(3), source_homography[n], -destination[n, 1] * source_homography[n]]])
                  for n in range(source.shape[0])]).reshape(2 * source.shape[0], 9)
    [_, _, V] = np.linalg.svd(A)
    h = V[-1,:]/V[-1,-1]
    h = np.array(h)
    return np.reshape(h ,(3 ,3 ))

def apply_homography(H, source):
    source_homography = np.hstack((source, np.ones((source.shape[0], 1))))
    destination =  source_homography @ H.T
    return (destination / destination[:,[2]])[:,0:2]

def getHomography(source , destination , good_points_matches):
    threshold = 1
    iterations = 50
    best_count = 0 
    H_final = np.zeros((3,3))
    best_dist = np.inf

    normalize_source_x , normalize_source_y , T_source_normalize = normalize_keypoints(source , good_points_matches)
    normalize_source = np.hstack((normalize_source_x.reshape(-1 ,1) , normalize_source_y.reshape(-1 ,1)))

    normalize_destination_x , normalize_destination_y , T_destination_normalize = normalize_keypoints(destination , good_points_matches)
    normalize_destination = np.hstack((normalize_destination_x.reshape(-1 ,1) , normalize_destination_y.reshape(-1 ,1)))

    for i in range(iterations): 
        rand_4 = np.random.randint( 0 , good_points_matches , size = 4)
        source_4 = np.vstack((normalize_source[rand_4[0] ,:] , normalize_source[rand_4[1] ,:] , normalize_source[rand_4[2] ,:] , normalize_source[rand_4[3] ,:] ))
        destination_4 = np.vstack((normalize_destination[rand_4[0] ,:] , normalize_destination[rand_4[1] ,:] , normalize_destination[rand_4[2] ,:] , normalize_destination[rand_4[3] ,:] ))
    
        H = estimate_homography( source_4 , destination_4 )
        destination_after_homography = apply_homography(H , normalize_source)

        # Computing L2 norm between newly computed and existing points
        dist = np.linalg.norm(destination_after_homography - normalize_destination , axis =1)
        dis_sum = np.sum(dist)

        inlier_find = np.where( dist<threshold , 1 , 0)
        count = np.sum(inlier_find)

    # Check condition to store the H matrix with max. inliers
        if count>best_count or (count==best_count and dis_sum<best_dist):
            best_count = count
            best_dist = dis_sum
            H_final = H

    # Denormalizing H matrix
    H_recapture = np.linalg.inv(T_destination_normalize) @ H_final @ T_source_normalize

    return H_recapture 

def normalize_keypoints(points, good_points_matches):
    x_mean = np.mean(points[: , 0 , 0])
    y_mean = np.mean(points[: , 0 , 1])

    original_xy = np.hstack((  points[: , 0 , 0].reshape(-1 ,1) , points[: , 0 , 1].reshape(-1 ,1)))
    std_check = np.std(original_xy)

    t_matrix = np.array([ [(np.sqrt(2)/std_check) , 0 , -(x_mean*np.sqrt(2)/std_check)] ,
                        [ 0 , (np.sqrt(2)/std_check) , -(y_mean*np.sqrt(2)/std_check)] ,
                        [ 0 , 0 , 1]    ] )

    points = np.vstack((  points[: , 0 , 0].reshape(1,-1), points[: , 0 , 1].reshape(1,-1) ,  np.ones((1 , good_points_matches))))

    points_norm = np.matmul(t_matrix, points)

    points_norm_x = points_norm[0 , :]
    points_norm_y = points_norm[1 , :] 

    return points_norm_x , points_norm_y , t_matrix

def plot_matches(images):
    img1, img2, img3= images

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_gray,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_gray,None)
    keypoints_3, descriptors_3 = sift.detectAndCompute(img3_gray,None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches1 = bf.match(descriptors_1, descriptors_2)
    matches1 = sorted(matches1, key=lambda x: x.distance)
    matched1_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches1[:50], None, flags=2)
    cv2.imshow("Img 1 and Img2 matches", matched1_image)
    cv2.imwrite("Output/Img_1_and_Img2_matches.jpg", matched1_image)
    cv2.waitKey(0)

    matches2 = bf.match(descriptors_2, descriptors_3)
    matches2 = sorted(matches2, key=lambda x: x.distance)
    matched2_image = cv2.drawMatches(img2, keypoints_2, img3, keypoints_3, matches2[:50], None, flags=2)
    cv2.imshow("Img 2 and Img3 matches", matched2_image)
    cv2.imwrite("Output/Img_2_and_Img3_matches.jpg", matched2_image)
    cv2.waitKey(0)