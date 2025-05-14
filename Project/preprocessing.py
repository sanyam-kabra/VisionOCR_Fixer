import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#function to set correct orientation for image
def correct_skew(image):
    """
    Takes an OpenCV image (numpy array) and returns the deskewed image.
    """

    # Step 1: Find coordinates of all non-zero pixels (text regions)
    coords = np.column_stack(np.where(image > 0))

    # Step 2: Compute angle of the minimum-area bounding box
    angle = cv2.minAreaRect(coords)[-1]

    # Step 3: Adjust the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Step 4: Get the image center and compute the rotation matrix
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Step 5: Rotate the image to correct skew
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # # Optional: Draw the angle (for debugging)
    # print(f"[INFO] Detected angle: {angle:.3f} degrees")

    return rotated

# Function to apply preprocessing steps
def preprocess_image(image):
    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Step 2: Remove Noise using Median Blur
    blurred = cv2.medianBlur(gray, 7)

    # Step 3: Apply Adaptive Thresholding (Binarization)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    
    # Step 4a: Dilation
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # # Step 4b: Erosion
    # eroded = cv2.erode(dilated, kernel, iterations=1)

    # # Skew Correction 
    # # deskewed_img = correct_skew(eroded)

    return dilated

# Function to process a single image (for image files)
def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Apply the preprocessing steps
    processed_img = preprocess_image(img)

    # Save or show the processed image
    cv2.imwrite('processed_image.png', processed_img)
    # cv2.imshow('Processed Image', processed_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return processed_img

# Function to process PDF pages (convert PDF to images and apply preprocessing)
def process_pdf(pdf_path):
    # Convert the PDF pages to images
    images = convert_from_path(pdf_path, dpi=300)

    # Loop through each page image and preprocess it
    for i, page in enumerate(images):
        # Convert PIL image to OpenCV format (NumPy array)
        open_cv_image = np.array(page)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Apply preprocessing steps
        processed_img = preprocess_image(open_cv_image)

        # Save or show the processed image (for each page)
        output_image_path = f'processed_page_{i + 1}.png'
        cv2.imwrite(output_image_path, processed_img)
        cv2.imshow(f'Processed Page {i + 1}', processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("PDF Processing Done!")

# Example Usage
# Process a single image
final_image = process_image('Test_image/Test_image3.png')  # Provide your image path here


# # Process a PDF
# process_pdf('sample_document.pdf')  # Provide your PDF path here

text = pytesseract.image_to_string(final_image)
print(text)