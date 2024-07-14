import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector

# Initialize the video capture and pose detector
cap = cv2.VideoCapture("Resource/video/myvideo.mp4")
detector = PoseDetector()

# Load the list of shirt images
shirtfolder = "Resource/shirts"
listshirt = os.listdir(shirtfolder)
print(listshirt)

# Define the ratio for the shirt's height to width
shirtratio = 581 / 440

# Define minimum dimensions for the shirt image
min_shirt_width = 250  # Increased minimum width for better visibility

while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Find the pose and positions
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    if lmList:
        # Get the coordinates for the left and right shoulders
        lm11 = lmList[11][0:3]  # Left shoulder
        lm12 = lmList[12][0:3]  # Right shoulder

        # Load the first shirt image
        imgshirt = cv2.imread(os.path.join(shirtfolder, listshirt[2]), cv2.IMREAD_UNCHANGED)

        if imgshirt is not None:
            # Calculate the width of the shirt based on the distance between shoulders
            shoulderWidth = int((lm12[0] - lm11[0]) * 1.2)  # Adjusted scaling factor
            shirtHeight = int(shoulderWidth * shirtratio)

            # Ensure the dimensions are above the minimum size
            shoulderWidth = max(min_shirt_width, shoulderWidth)
            shirtHeight = max(int(min_shirt_width * shirtratio), shirtHeight)

            imgshirt = cv2.resize(imgshirt, (shoulderWidth, shirtHeight))

            # Calculate the position to place the shirt
            xOffset = lm11[0] - shoulderWidth // 2
            yOffset = lm11[1]  # Start from the left shoulder and move down

            # Move the shirt further down and to the left
            yOffset -= 30  # Adjust this value to move the shirt lower
            xOffset -= 70  # Adjust this value to move the shirt to the left

            # Print debug information
            ''' print(f"Shoulder Width: {shoulderWidth}")
            print(f"Shirt Height: {shirtHeight}")
            print(f"lm11: {lm11}")
            print(f"lm12: {lm12}")
            print(f"Initial xOffset: {xOffset}")
            print(f"Initial yOffset: {yOffset}")'''

            # Ensure the coordinates are within the bounds of the original image
            xOffset = max(0, min(xOffset, img.shape[1] - shoulderWidth))
            yOffset = max(0, min(yOffset, img.shape[0] - shirtHeight))

            #Print out the final values for debugging
            '''  print(f"Final xOffset: {xOffset}")
            print(f"Final yOffset: {yOffset}")
            print(f"Placing shirt at: ({xOffset}, {yOffset}), Size: ({shoulderWidth}, {shirtHeight})")'''

            # Overlay the shirt image on the original image
            try:
                img = cvzone.overlayPNG(img, imgshirt, [xOffset, yOffset])
            except Exception as e:
                print(f"Error overlaying PNG: {e}")
            else:
                print("Failed to load shirt image.")

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
