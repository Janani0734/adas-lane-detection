import cv2
import numpy as np
import time

def process_frame(frame):
    # 1. Image Preprocessing Matrix
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    
    # 2. Region of Interest Masking
    height, width = canny.shape
    polygon_points = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.7), int(height * 0.6)),
        (int(width * 0.3), int(height * 0.6))
    ]], dtype=np.int32)
    
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, polygon_points, 255)
    masked_edges = cv2.bitwise_and(canny, mask)
    
    # 3. Hough Line Transform Math
    # Extracts linear line vectors from our isolated white pixels
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20, 
        minLineLength=20, 
        maxLineGap=30
    )
    
    # 4. Rendering the Lines over the Original Video
    # Create an empty blank frame to draw our lane overlays on
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw a solid red line (BGR: 0, 0, 255) with a thickness of 5 pixels
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
            
    # Merge the original color frame with our red line overlay
    final_output = cv2.addWeighted(frame, 0.8, line_image, 1.0, 0.0)
    
    return final_output

def main():
    video_path = "highway_clip.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    frame_count = 0
    start_time = time.time()

    print("Running full autonomous pipeline... Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Run our completed perception pipeline
        lane_tracking_output = process_frame(frame)
        
        # Display the final composite video window
        cv2.imshow("Wipro ADAS Demo: Real-Time Lane Tracking", lane_tracking_output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    actual_fps = frame_count / elapsed_time

    print("\n--- FINAL VERIFIED ADAS BENCHMARK ---")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Elapsed Time:     {elapsed_time:.2f} seconds")
    print(f"Final Production Speed: {actual_fps:.2f} FPS")
    print("-------------------------------------")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()