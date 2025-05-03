import cv2
import numpy as np
import time

static_image = cv2.imread("""your image path""")
static_image = cv2.resize(static_image, (288, 352))
if static_image is None:
    raise FileNotFoundError("Static image not found at specified path")

static_image_gray = cv2.cvtColor(static_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,
    edgeThreshold=15,
    patchSize=31
)

FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=2
)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

kp1, des1 = orb.detectAndCompute(static_image_gray, None)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = []
match_counts = []
inlier_ratios = []

while True:
    start_time = time.time()
    
    ret, live_frame = cap.read()
    if not ret:
        break

    live_frame = cv2.resize(live_frame, (288, 352))
    live_frame = cv2.flip(live_frame, 1)
    live_frame_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(live_frame_gray, None)
    
    if des2 is None or des1 is None:
        continue

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    match_counts.append(len(good_matches))
    
    h1, w1 = static_image.shape[:2]
    h2, w2 = live_frame.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = static_image
    vis[:h2, w1:w1+w2] = live_frame
    
    inlier_count = 0
    reproj_error = None
    
    if len(good_matches) >= 10:  #increased for better stabality
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        #RANSAC Params
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, confidence=0.99, maxIters=2000)
        
        if H is not None and not np.isnan(H).any():
            inlier_count = np.sum(mask)
            inlier_ratio = inlier_count / len(good_matches) if len(good_matches) > 0 else 0
            inlier_ratios.append(inlier_ratio)
            
            if inlier_count > 0:
                reproj_errors = []
                for i, m in enumerate(good_matches):
                    if mask[i][0]:
                        src_pt = np.float32([kp1[m.queryIdx].pt]).reshape(-1,1,2)
                        dst_pt = np.float32([kp2[m.trainIdx].pt]).reshape(-1,1,2)
                        transformed_pt = cv2.perspectiveTransform(src_pt, H)
                        error = np.sqrt(np.sum((transformed_pt - dst_pt)**2))
                        reproj_errors.append(float(error))

                reproj_error = np.mean(reproj_errors) if reproj_errors else None
            
            h, w = static_image_gray.shape
            corners = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
            
            try:
                transformed_corners = cv2.perspectiveTransform(corners, H)
                shifted_corners = transformed_corners + np.float32([w1, 0])
                cv2.polylines(vis, [np.int32(shifted_corners)], True, (0,255,0), 3)
                
                #draw only the matching lines (not circles) for inliers
                for i, m in enumerate(good_matches):
                    if mask[i][0]:
                        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
                        pt2 = tuple(map(int, (kp2[m.trainIdx].pt[0] + w1, kp2[m.trainIdx].pt[1])))
                        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            except cv2.error as e:
                print(f"Homography error: {str(e)}")
    else:
        max_to_draw = min(len(good_matches), 10)
        for i in range(max_to_draw):
            m = good_matches[i]
            pt1 = tuple(map(int, kp1[m.queryIdx].pt))
            pt2 = tuple(map(int, (kp2[m.trainIdx].pt[0] + w1, kp2[m.trainIdx].pt[1])))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

    frame_time = time.time() - start_time
    frame_times.append(frame_time)
    
    recent_times = frame_times[-30:]
    avg_fps = 1.0 / (sum(recent_times) / len(recent_times)) if recent_times else 0
    
    recent_matches = match_counts[-30:]
    avg_matches = sum(recent_matches) / len(recent_matches) if recent_matches else 0
    
    recent_inliers = inlier_ratios[-30:]
    avg_inlier_ratio = sum(recent_inliers) / len(recent_inliers) if recent_inliers else 0
    


    cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"Matches: {len(good_matches)} (avg: {avg_matches:.1f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if inlier_count > 0:
        inlier_percent = 100 * inlier_count / len(good_matches)
        cv2.putText(vis, f"Inliers: {inlier_count} ({inlier_percent:.1f}%)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if reproj_error is not None:
        cv2.putText(vis, f"Reproj Error: {reproj_error:.2f}px", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Feature Matching", vis)

    
    if cv2.waitKey(1) & 0xFF == 27:
        break

if frame_times:
    print(f"Average FPS: {1.0 / (sum(frame_times) / len(frame_times)):.2f}")
if match_counts:
    print(f"Average matches: {sum(match_counts) / len(match_counts):.2f}")
if inlier_ratios:
    print(f"Average inlier ratio: {sum(inlier_ratios) / len(inlier_ratios):.2f}")

cap.release()
cv2.destroyAllWindows()
