import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 23, 0.001)
N = 9
M = 6
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(M,5,0)
objp = np.zeros((M*N,3), np.float32)
objp[:,:2] = np.mgrid[0:N,0:M].T.reshape(-1,2)

print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibration_images_webcam/*.jpg')
print(f"found {len(images)} images")
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (N,M), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("found chess board in", fname)
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (N,M), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)
    else:
        print(f"fail to find chessboard corner in {fname}")

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx.tolist())

print(dist.tolist())

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )