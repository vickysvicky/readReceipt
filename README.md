# readReceipt

lambda_function.py
- add S3 PUT trigger to function
- get uploaded image, process it, and parse the receipt
- processed image is uploaded to same S3 bucket
- return parsed result as response
- image preprocessing: 
    - detect receipt, crop and warp image to top down view
    - convert colour to grayscale
    - Gaussian blurring
    - Canny edge detection
- ocr:
    - parse receipt with tesseract

apikey.txt
- if uses free ocr-api in lambda_function.py, paste the key here

lambda-layers-build
- from https://github.com/amtam0/lambda-tesseract-api
- builds opencv-python (with numpy), pytesseract, tesseract, imutils zip files
- to be uploaded as lambda layers
- modified build_py37_pkgs.sh to also build imutils package

