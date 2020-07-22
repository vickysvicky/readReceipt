# readReceipt
for AWS Serverless Lambda Function
running in Python 3.7
Parse receipt when it is uploaded to designated S3 Bucket

lambda_function.py
- add S3 Bucket PUT trigger to function
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
- builds opencv-python (with numpy), pytesseract, tesseract, imutils zip files
- to be uploaded as lambda layers
- modified build_py37_pkgs.sh to also build imutils package


# Reference
[image preprocessing] https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
[building lambda layers] https://github.com/amtam0/lambda-tesseract-api
