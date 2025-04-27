from PIL import Image
import pytesseract

# Tell pytesseract where Tesseract-OCR is installed
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the image
image = Image.open('image2.png')

# OCR
text = pytesseract.image_to_string(image)

print(text)
