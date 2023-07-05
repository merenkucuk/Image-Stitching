import os
import cv2
import random
import numpy as np

# Path of img directory
IMG_DIR = r'project'


def homography(point):
    homographyArr = []
    for i in range(len(point)):
        x = point[i][0]
        y = point[i][1]
        a = point[i][2]
        b = point[i][3]

        homographyArr.append([x, y, 1,
                              0, 0, 0,
                              -a * x, -a * y, -a])

        homographyArr.append([0, 0, 0,
                              x, y, 1,
                              -b * x, -b * y, -b])

    homographyArr = np.array(homographyArr)
    svdValue = np.linalg.svd(homographyArr)
    vh = svdValue[2]
    vhArray = np.array(vh)
    vhLength = len(vhArray)
    L = vhArray[vhLength - 1] / vhArray[vhLength - 1, -1]
    homographyResult = L.reshape(3, 3)

    return homographyResult


def n_length_combo(lst, n):
    if n == 0:
        return [[]]

    l = []
    for i in range(0, len(lst)):

        m = lst[i]
        remLst = lst[i + 1:]

        for p in n_length_combo(remLst, n - 1):
            l.append([m] + p)

    return l


def ransac(poc):
    assert (len(poc) > 40)
    # number of inliers
    bestS = 0
    # inliers arr
    bestArr = []
    # best points
    bestPoints = poc[:40]

    # 4 count best points
    matchBestP = list(n_length_combo(bestPoints, 4))

    # Shuffle
    matchBestP[:] = random.sample(matchBestP, len(matchBestP))

    # Process Ransac
    for matches in matchBestP[:5000]:

        H = homography(matches)

        inliers = []
        count = 0

        # Find inliers
        for i in range(len(bestPoints)):
            src = np.full((3, 1), 1.)
            tgt = np.full((3, 1), 1.)
            src[0, 0] = bestPoints[i][0]
            src[1, 0] = bestPoints[i][1]
            tgt[0, 0] = bestPoints[i][2]
            tgt[1, 0] = bestPoints[i][3]

            # Become features from homography
            tgt_hat = np.matmul(H, src)

            if tgt_hat[len(tgt_hat) - 1][0] == 0:
                continue

            else:
                # Scale unity plane
                tgt_hat /= tgt_hat[len(tgt_hat) - 1][0]
                # Check if it is inlier
                if np.linalg.norm(tgt_hat - tgt) < 4:
                    count += 1
                    inliers.append(bestPoints[i])

        # bucket of best inliers
        if count <= bestS:
            continue

        else:
            bestS = count
            bestArr = inliers

    best_H = homography(bestArr)

    return best_H


def matchFeatures(src, tgt, nfeatures=1000):
    orb = cv2.ORB_create(nfeatures=nfeatures, scoreType=cv2.ORB_FAST_SCORE)
    kp1, des1 = orb.detectAndCompute(src, None)
    kp2, des2 = orb.detectAndCompute(tgt, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    bestPoints = []
    for i in range(len(matches)):
        x1y1 = np.float32(kp1[matches[i].queryIdx].pt)
        x2y2 = np.float32(kp2[matches[i].trainIdx].pt)
        feature = list(map(int, list(x1y1) + list(x2y2) + [matches[i].distance]))
        bestPoints.append(feature)

    bestPoints = np.array(bestPoints)

    """
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY), None)

    # Using Brute Force matcher to find matches.
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)
    
  
    kp = Sift.detect(src, None)
    img = cv2.drawKeypoints(src, kp, src)
    plt.imshow(img), plt.show()

    # Applytng ratio test and filtering out the good matches.

    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])
            
    return GoodMatches
    """

    """
    # displaying the image with drawing matches
    img3 = cv2.drawMatches(src, kp1, tgt, kp2, matches[:20], tgt, flags=2)
    plt.imshow(img3), plt.show()
    # displaying the image with keypoints as the output on the screen
    img_1 = cv2.drawKeypoints(src, kp1, src)
    plt.imshow(img_1), plt.show()
    """

    """
    Sift = cv2.SIFT_create()
    kp = Sift.detect(src, None)
    kp5 = Sift.detect(tgt, None)
    
    # displaying the image with keypoints as the output on the screen
    img = cv2.drawKeypoints(src, kp, src)
    plt.imshow(img), plt.show()
    
    img3 = cv2.drawMatches(src, kp, tgt, kp5, matches[:20], tgt, flags=2)
    plt.imshow(img3), plt.show()
    """

    return bestPoints


def warpImages(src, homography, dst):
    src = np.array(src)
    inputH = src.shape[0]
    inputW = src.shape[1]

    # Checking if image needs to be warped or not
    if homography == None:
        dst[H // 2:H // 2 + inputH, W // 2:W // 2 + inputW] = src

    else:
        # Calculating net homography
        t = homography
        homography = np.eye(3)
        for i in range(len(t)):
            homography = np.matmul(t[i], homography)

        # Finding bounding box
        pts = np.array([[0, 0, 1], [inputW, inputH, 1], [inputW, 0, 1], [0, inputH, 1]]).T
        borders = (np.matmul(homography, pts.reshape(3, -1)).reshape(pts.shape))
        borders /= borders[-1]
        borders = (borders + np.array([W // 2, H // 2, 0])[:, np.newaxis]).astype(int)
        h_min, h_max = np.min(borders[1]), np.max(borders[1])
        w_min, w_max = np.min(borders[0]), np.max(borders[0])

        # Filling the bounding box in imgout
        h_inv = np.linalg.inv(homography)
        for i in (range(h_min, h_max + 1)):
            for j in range(w_min, w_max + 1):

                if (0 <= i < H and 0 <= j < W):
                    # Calculating image cordinates for src
                    u, v = i - H // 2, j - W // 2
                    src_j, src_i, scale = np.matmul(h_inv, np.array([v, u, 1]))
                    src_i, src_j = int(src_i / scale), int(src_j / scale)

                    # Checking if cordinates lie within the image
                    if (0 <= src_i < inputH and 0 <= src_j < inputW):
                        dst[i, j] = src[src_i, src_j]

    # Creating a alpha mask of the transformed image
    mask = np.sum(dst, axis=2).astype(bool)
    return dst, mask


def laplacianPyramidBlend(images, masks):
    g_pyramids = {}
    l_pyramids = {}
    W = images[0].shape[1]

    for i in range(len(images)):
        # Gaussian Pyramids
        G = images[i].copy()
        g_pyramids[i] = [G]
        for k in range(5):
            G = cv2.pyrDown(G)
            g_pyramids[i].append(G)

        # Laplacian Pyramids
        l_pyramids[i] = [G]
        for j in range(len(g_pyramids[i]) - 2, -1, -1):
            G_up = cv2.pyrUp(G)
            G = g_pyramids[i][j]
            L = cv2.subtract(G, G_up)
            l_pyramids[i].append(L)

    common_mask = masks[0].copy()
    common_pyramids = [l_pyramids[0][i].copy() for i in range(len(l_pyramids[0]))]

    ls_ = None

    for i in range(1, len(images)):
        y1, x1 = np.where(common_mask == 1)
        y2, x2 = np.where(masks[i] == 1)

        if np.max(x1) > np.max(x2):
            left_py = l_pyramids[i]
            right_py = common_pyramids

        else:
            left_py = common_pyramids
            right_py = l_pyramids[i]

        mask_intersection = np.bitwise_and(common_mask, masks[i])

        if True in mask_intersection:
            # If images blend, we need to find the center of the overlap
            y, x = np.where(mask_intersection == 1)
            x_min, x_max = np.min(x), np.max(x)

            # We get the split point
            split = ((x_max - x_min) / 2 + x_min) / W

            # Finally we add the pyramids
            LS = []
            for la, lb in zip(left_py, right_py):
                cols = la.shape[1]
                ls = np.hstack((la[:, 0:int(split * cols)], lb[:, int(split * cols):]))
                LS.append(ls)

        else:
            LS = []
            for la, lb in zip(left_py, right_py):
                ls = la + lb
                LS.append(ls)

        # Reconstructing the image
        ls_ = LS[0]
        for j in range(1, 6):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[j])

        # Preparing the common image for next image to be added
        common_image = ls_
        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)
        common_pyramids = LS

    return ls_


valid_images = [".jpg", ".png", ".jpeg"]
Images = []
A = 0
B = 0
C = 0
for f in os.listdir(IMG_DIR):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    img = cv2.imread(os.path.join(IMG_DIR, f))
    img = cv2.resize(img, (640, 480))
    x, y, z = img.shape
    A = x
    B = y
    C = z
    Images.append(img)
N = len(Images)
H, W, C = np.array((A, B, C)) * [3, N, 1]  # Finding shape of final image

# Image Template for final image
img_f = np.zeros((H, W, C),dtype='uint8')
img_outputs = []
masks = []

img, mask = warpImages(Images[N // 2], None, img_f.copy())

img_outputs.append(img)
masks.append(mask)
left_H = []
right_H = []

for i in range(1, len(Images) // 2 + 1):
    try:
        poc = matchFeatures(Images[N // 2 + i], Images[N // 2 + (i - 1)])
        right_H.append(ransac(poc))
        img, mask = warpImages(Images[N // 2 + i], right_H[::-1], img_f.copy())
        img_outputs.append(img)
        masks.append(mask)
    except:
        pass

    try:
        poc = matchFeatures(Images[N // 2 - i], Images[N // 2 - (i - 1)])
        left_H.append(ransac(poc))
        img, mask = warpImages(Images[N // 2 - i], left_H[::-1], img_f.copy())
        img_outputs.append(img)
        masks.append(mask)
    except:
        pass

uncropped = laplacianPyramidBlend(img_outputs, masks)
# Creating a mask of the panaroma
mask = np.sum(uncropped, axis=2).astype(bool)
# Finding appropriate bounding box
yy, xx = np.where(mask == 1)
# Croping and saving
final = uncropped[np.min(yy):np.max(yy), np.min(xx):np.max(xx)]
cv2.imwrite("StitchedImage.jpg", final)
