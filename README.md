# readReceipt
for AWS Serverless Lambda Function <br />
running in Python 3.7 <br />
Parse receipt when it is uploaded to designated S3 Bucket <br />

## lambda_function.py ##
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
<br />
## apikey.txt ##
- if uses free ocr-api in lambda_function.py, paste the key here
<br />
## lambda-layers-build ##
- uses Docker to build opencv-python (with numpy), pytesseract, tesseract, imutils zip files
- modifiy build_py37_pkgs.sh to also build desired packages
- runtime: python 3.7
- zip files to be uploaded as lambda layers
- run `bash build_py37_pkgs.sh`
- run `bash build_tesseract4.sh` 
<br />
<br />
# Reference
[image preprocessing] https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/ <br />
[building lambda layers] https://github.com/amtam0/lambda-tesseract-api
