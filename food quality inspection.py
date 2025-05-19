

import cv2
import numpy as np

def analyze_color_variance(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    variance = np.var(hist)
    return variance

def analyze_texture_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = lap.var()
    return sharpness

def classify_food(color_var, sharpness):
  
    if sharpness < 30 or color_var > 0.045:
        return "Reject", (0, 0, 255) 
    elif sharpness > 80 and color_var < 0.02:
        return "Premium", (0, 255, 0) 
    else:
        return "Acceptable", (0, 255, 255)

def inspect_food(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(" Image not found.")
        return

    image = cv2.resize(image, (600, 400))

    color_var = analyze_color_variance(image)
    sharpness = analyze_texture_sharpness(image)

    result, color = classify_food(color_var, sharpness)

    print(f" Image: {image_path}")
    print(f" Texture Sharpness: {sharpness:.2f}")
    print(f" Color Variance: {color_var:.5f}")
    print(f" Quality: {result}")

    cv2.putText(image, f"Quality: {result}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Food Quality Inspection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


inspect_food("dry.jpg")
