import cv2
import numpy as np

def computeDes(img1, img2):

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize the feature detector and extractor (e.g., SIFT)
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    return (keypoints1, descriptors1), (keypoints2, descriptors2)

def feature_matching(keypoints1, descriptors1, keypoints2, descriptors2):

    # Initialize the feature matcher using FLANN matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Select the top N matches
    num_matches = 50
    # Match descriptors using FLANN
    matches_flann = flann.match(descriptors1, descriptors2)

    # Sort the matches by distance (lower is better)
    matches= sorted(matches_flann, key=lambda x: x.distance)[:num_matches]

    # Draw the top N matches
    # image_matches_flann = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_flann[:num_matches], None)

    # Extract matching keypoints
    src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    return src_points, dst_points

def findHomography(src_points, dst_points):
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return homography

def stitImage(right_img, left_img):

    (keypoints1, descriptors1), (keypoints2, descriptors2) = computeDes(right_img, left_img)

    src_points, dst_points = feature_matching(keypoints1, descriptors1, keypoints2, descriptors2)

    homography = findHomography(src_points, dst_points)

    h1, w1 = right_img.shape[:2]
    h2, w2 = left_img.shape[:2]
    result = cv2.warpPerspective(right_img, homography, (w1+w2, max(h1,h2)))
    result[0:h2, 0:w2] = left_img

    return result

def crop(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tạo mặt nạ nhị phân (binary mask) để xác định các vùng không đen
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Tìm các đường viền của vùng không đen
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tìm hộp bao phủ tối thiểu cho vùng không đen
    x, y, w, h = cv2.boundingRect(contours[0])

    # Cắt ảnh theo hộp bao phủ
    cropped_image = img[y:y+h, x:x+w]

    return cropped_image

l_img = cv2.imread("images/left.png")
r_img = cv2.imread("images/right.png")

if __name__ == "__main__":

    result = stitImage(r_img, l_img)
    cropped_image = crop(result)
    cv2.imshow("Stit Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()