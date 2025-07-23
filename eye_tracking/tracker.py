import cv2
import os
import time
import numpy as np

def track_eye(video_source=1):# 0 for default webcam, 1 for external camera
    # Load the pre-trained Haar Cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if eye_cascade.empty():
        print("Error: Could not load eye cascade classifier")
        return
    
    # Create directory for captured eyes
    os.makedirs('captured_eyes', exist_ok=True)
    
    # Initialize video capture (removed CAP_DSHOW for compatibility)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Try to set camera properties (may not work on all cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    capture_count = 0
    tracking_active = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for intuitive viewing
            frame = cv2.flip(frame, 1)
            
            # Create clean copy for saving
            clean_frame = frame.copy()
            
            if tracking_active:
                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect eyes with optimized parameters
                eyes = eye_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(40, 40)
                )
                
                # Draw visual feedback
                for (x, y, w, h) in eyes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    center = (x + w//2, y + h//2)
                    cv2.line(frame, (center[0]-10, center[1]), (center[0]+10, center[1]), (0, 0, 255), 1)
                    cv2.line(frame, (center[0], center[1]-10), (center[0], center[1]+10), (0, 0, 255), 1)
                
                # Display info
                cv2.putText(frame, f"Eyes detected: {len(eyes)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "'c': Capture | 'p': Pause | 'q': Quit", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Eye Tracking System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                if 'eyes' in locals() and len(eyes) > 0:
                    timestamp = int(time.time())
                    for i, (x, y, w, h) in enumerate(eyes):
                        # Save clean eye image
                        eye_img = clean_frame[y:y+h, x:x+w]
                        filename = f"captured_eyes/eye_{timestamp}_{i}.png"
                        cv2.imwrite(filename, eye_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])  # Balanced compression
                        print(f"Saved {filename}")
                    capture_count += 1
            elif key == ord('p'):
                tracking_active = not tracking_active
                print("Tracking", "resumed" if tracking_active else "paused")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Session ended. Captured {capture_count} eye images.")

if __name__ == "__main__":
    print("Eye Tracking System")
    print("Instructions:")
    print(" - 'c': Capture eye image (without markers)")
    print(" - 'p': Pause/resume tracking")
    print(" - 'q': Quit program")
    track_eye()