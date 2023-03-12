## Image Stitching based on Homography(RANSAC)

Run using the following code:

    python3 main.py
    
The following code takes the following steps to perform image stitching:
1. Finding SIFT Features using cv2 inbuilt function

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/224551616-4ba24244-cef6-4c7f-a40b-1b1afd621458.jpg" width="300" height="300" />
</p>

2. Finding Matches between pairs (img1, img2) and (img2, img3). Image 2 being the central image is considered as the reference point to warp around.
3. Here, I have used BF matcher for matching as it is one of the reliable and commonly used techniques.
4. Image 1 and Image 2 matches
<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/224549771-09c5887e-c8b0-4f10-b18c-447a050cae9f.jpg" width="500" height="300" />
</p>

  5. Image 2 and Image 3 matches
<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/224550346-207feff9-3d5f-4c51-91db-dcf0d3c12664.jpg" width="500" height="300" />
</p>

6. Next step involves finding homography between each of these using RANSAC (function present in utils.py)

    a. Homography estimation begins by normalizing points, so we are left with only 8 variables instead of original 9 in a 3x3 matrix.
    
    b. For RANSAC, we take pairs of random 4 points and try to fit the curve. 
    
    c. Running RANSAC for a number of cycles(predefined-1000), we obtain the homography with the maximum number of inliers.
    
    d. Inliers are calculated based on their distance from the plotted curve and we threshold on the distance to call it an inlier or an outlier.
    
    e. Then, we denormalize the homography once we have it our final homography from RANSAC.
    
    f. Post this, we can simply apply the homography and obtain our estimated destination coordinates.
    
    g. We can now use the estimated destination points and the original destination matches to obtain the error in homography.
    
7. The final homographies comes out to be:

    Best H for img 1 and img 2

   $$
   \left(\begin{array}{cc} 
     1.34885825e+00 & 7.51001649e-02 & 3.32980816e+02\\
     1.50896706e-01 & 1.02700460e+00 & -7.87776055e+01\\
     4.09395544e-04 & 7.70129860e-05 & 8.19723455e-01\\
    \end{array}\right)
    $$
    
    Best H for img 2 and img 3

   $$
   \left(\begin{array}{cc} 
     5.01016563e-01 & -5.57994647e-02 & 9.07358831e+02\\
    -2.01712078e-01 & 9.66437829e-01 & 6.36870202e+01\\
    -4.90257479e-04 & -6.04159853e-05 & 1.12176388e+00\\
    \end{array}\right)
    $$
7. We can use the homography to warp 2 images using the inbuild function - cv2.warpPerspective(src_img, homography, shape_of_final_image)

8. Merge the warped and original image

9. The final stitched output looks like this:

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/224551313-8f0c84a7-425c-4de7-86f3-00714aa252a2.jpg" width="800" height="300" 
</p>
