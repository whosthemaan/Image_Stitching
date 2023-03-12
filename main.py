import cv2
import numpy as np
from utils import load_images, good_points_matches, getHomography, plot_matches

def stitch_images(images):  
    img1, img2, img3= images

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    # padding middle image(2nd image from left)
    middle_img = np.zeros((img1_gray.shape[0] , img1_gray.shape[1]+img2_gray.shape[1]) , dtype = np.uint8)
    middle_img[: , img1_gray.shape[1]:] = img2_gray

    # For Image 1 and 2
    src , dst , match_count = good_points_matches(img1_gray , middle_img , True)
    H_1 = getHomography(src , dst , match_count)

    print(f"Best H for img 1 and img 2\n{H_1}\n")
    dst = cv2.warpPerspective(img1 , H_1,(img1.shape[1] + img2.shape[1], img2.shape[0]))
    dst[:, img2.shape[1]:] = img2

    src_2 , dst_2 , match_count_2 = good_points_matches(dst, img3_gray, False)
    H_2 = getHomography(src_2 , dst_2 , match_count_2)

    print(f"Best H for first_warp and img 3\n{H_2}\n")

    merged = cv2.warpPerspective(img3 , H_2,(dst.shape[1] + img3_gray.shape[1], img3_gray.shape[0]))
    merged[:, :dst.shape[1]] = dst

    cv2.imwrite("Output/Stitched_Image.jpg", merged)
    cv2.imshow("Stitched_Image", merged)
    cv2.waitKey(0)

def main():
    images = load_images()
    print("Starting Image Stitching Procedure\n")
    plot_matches(images)
    stitch_images(images)

if __name__=="__main__":
    main()

