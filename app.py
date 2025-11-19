import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imagehash
import os
import shutil

# =========================================================
# HELPER: SAVE UPLOADS TO DISK
# =========================================================
# OpenCV reads from paths best, so we save uploads temporarily
def save_uploaded_file(uploaded_file):
    try:
        # Create a temp dir if not exists
        if not os.path.exists("temp_images"):
            os.makedirs("temp_images")
            
        file_path = os.path.join("temp_images", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# =========================================================
# YOUR CORE LOGIC (Adapted for Streamlit)
# =========================================================

def create_hash(image_path, hash_type='ahash', hash_size=8):
    try:
        # We use the path here because your logic expects it
        img = Image.open(image_path).convert('L')
        
        if hash_type == 'ahash':
            return imagehash.average_hash(img, hash_size=hash_size)
        elif hash_type == 'dhash':
            return imagehash.dhash(img, hash_size=hash_size)
        elif hash_type == 'phash':
            return imagehash.phash(img, hash_size=hash_size)
        elif hash_type == 'whash':
            return imagehash.whash(img, hash_size=hash_size)
    except Exception as e:
        st.error(f"Hashing Error: {e}")
        return None

def compare_hashes(hash1, hash2, hash_type, threshold):
    if hash1 is None or hash2 is None:
        return False, None
    
    distance = hash1 - hash2
    is_match = distance <= threshold
    
    # Visual output for the UI
    status = "MATCH" if is_match else "NO MATCH"
    color = "green" if is_match else "red"
    st.markdown(f"**{hash_type.upper()}**: Distance {distance} (Threshold {threshold}) → :{color}[{status}]")
    
    return is_match, distance

def feature_match_with_ransac(img_path_original, img_path_altered, min_inlier_count):
    st.subheader("Stage 2: ORB Feature Matching + RANSAC")
    
    img_original = cv2.imread(img_path_original, cv2.IMREAD_GRAYSCALE)
    img_altered = cv2.imread(img_path_altered, cv2.IMREAD_GRAYSCALE)
    
    if img_original is None or img_altered is None:
        st.error("Could not load images for ORB.")
        return 0, None

    orb = cv2.ORB_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, des1 = orb.detectAndCompute(img_original, None)
    kp2, des2 = orb.detectAndCompute(img_altered, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        st.warning("Insufficient keypoints for RANSAC.")
        return 0, None

    raw_matches = matcher.match(des1, des2)
    
    if len(raw_matches) < 4:
        st.warning("Not enough raw matches found.")
        return 0, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in raw_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in raw_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers_count = np.sum(mask) if mask is not None else 0

    col1, col2 = st.columns(2)
    col1.metric("Raw Matches", len(raw_matches))
    col2.metric("Verified Inliers", int(inliers_count), delta_color="normal" if inliers_count < min_inlier_count else "inverse")

    if inliers_count >= min_inlier_count:
        st.success(f"✅ RANSAC Verified! Found {int(inliers_count)} consistent features.")
    else:
        st.error(f"❌ RANSAC Rejected. Found {int(inliers_count)} matches (Needed {min_inlier_count}).")

    return inliers_count, M

# =========================================================
# STREAMLIT UI LAYOUT
# =========================================================

st.title("🖼️ Image Matcher Server")
st.write("Upload two images to verify if they are the same using Perceptual Hashing and Geometric Feature Matching.")

# --- Sidebar for Config ---
st.sidebar.header("Settings")
HASH_THRESHOLD = st.sidebar.slider("Hash Threshold", 0, 20, 8)
MIN_INLIER_COUNT = st.sidebar.slider("RANSAC Inlier Threshold", 4, 50, 8)

# --- File Uploads ---
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Original Image", type=['png', 'jpg', 'jpeg'])
    if file1: st.image(file1, caption="Original", use_container_width=True)
with col2:
    file2 = st.file_uploader("Altered/Target Image", type=['png', 'jpg', 'jpeg'])
    if file2: st.image(file2, caption="Target", use_container_width=True)

# --- Execution Button ---
if file1 and file2:
    if st.button("Run Comparison"):
        # Save files to disk so OpenCV can read them
        path1 = save_uploaded_file(file1)
        path2 = save_uploaded_file(file2)

        if path1 and path2:
            st.divider()
            
            # --- STAGE 1: HASHING ---
            st.subheader("Stage 1: Quick Hash Scan")
            
            h1_a = create_hash(path1, 'ahash')
            h2_a = create_hash(path2, 'ahash')
            match_a, _ = compare_hashes(h1_a, h2_a, 'aHash', HASH_THRESHOLD)

            h1_d = create_hash(path1, 'dhash')
            h2_d = create_hash(path2, 'dhash')
            match_d, _ = compare_hashes(h1_d, h2_d, 'dHash', HASH_THRESHOLD)
            
            h1_w = create_hash(path1, 'whash')
            h2_w = create_hash(path2, 'whash')
            match_w, _ = compare_hashes(h1_w, h2_w, 'wHash', HASH_THRESHOLD)

            # Logic to proceed
            if match_a or match_d or match_w:
                st.success("✅ Quick Match Found via Hashing.")
            else:
                st.info("No quick match. Proceeding to deep feature scan...")
                feature_match_with_ransac(path1, path2, MIN_INLIER_COUNT)

            # Cleanup temp files
            if os.path.exists("temp_images"):
                shutil.rmtree("temp_images")