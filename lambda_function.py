import json
import urllib
import boto3
import botocore

#   image preprocessing from :-
#   https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
#   lambda layer creation from :-
#   https://medium.com/analytics-vidhya/build-tesseract-serverless-api-using-aws-lambda-and-docker-in-minutes-dd97a79b589b

import cv2
# import requests     # for calling free ocr api
import imutils
import pytesseract
import numpy as np
# from skimage.filters import threshold_local       # exceed size limit


def order_points(pts):
    """
    take in 4 coordinates and organize them into
    top left, top right, bottom right, bottom left
    :param pts: 4 tuples of xy coordinates
    :return: (array) sorted 4 tuples of xy coordinates
    """

    #   top left, right, bottom right, left
    shape = np.zeros((4, 2), dtype='float32')
    #   top left has smallest x y sum
    #   bottom right has biggest x y sum
    s = pts.sum(axis=1)
    shape[0] = pts[np.argmin(s)]
    shape[2] = pts[np.argmax(s)]
    #   top right has smallest x y difference
    #   bottom left biggest x y difference
    d = np.diff(pts, axis=1)
    shape[1] = pts[np.argmin(d)]
    shape[3] = pts[np.argmax(d)]

    return shape


def transform_four_points(img, pts):
    """
    transform given image to top down view given corner coordinates of receipt
    :param img: (string) image path
    :param pts: 4 tuples of xy coordinates of the corners
    :return: top down view of receipt
    """
    #   get coordinates sorted
    shape = order_points(pts)
    (tl, tr, br, bl) = shape

    w1 = np.sqrt(((br[0]-bl[0]) ** 2) + ((br[1]-bl[1]) ** 2))
    w2 = np.sqrt(((tr[0]-tl[0]) ** 2) + ((tr[1]-tl[1]) ** 2))
    wM = max(int(w1), int(w2))

    h1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    h2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    hM = max(int(h1), int(h2))

    dim = np.array(([0, 0], [wM-1, 0], [wM-1, hM-1], [0, hM-1]), dtype='float32')

    #   get perspective transform matrix
    M = cv2.getPerspectiveTransform(shape, dim)

    #   apply M on img
    warped = cv2.warpPerspective(img, M, (wM, hM))

    return warped


def get_scanned(img, threshold=False, blur=False):
    """
    take in an image and save scanned version to /tmp/ folder
    :param img: (string) image name in 'receipt' folder
    :return: (boolean) if successfully transformed/scanned
    """
    #   Edge Detection
    #   read image
    image = cv2.imread(img)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)  # 500 pixels height

    #   convert to grayscale, blur it, and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # rgb to grayscale
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # gaussian blurring remove high freq noise
    edge = cv2.Canny(gray, 75, 200)  # canny edge detection

    # #   show edge of receipt
    # cv2.imshow('Receipt', image)
    # cv2.imshow('Edge', edge)
    # cv2.waitKey(0)

    #   Find Contour
    ct = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ct = imutils.grab_contours(ct)
    ct = sorted(ct, key=cv2.contourArea, reverse=True)[:5]  # keep first 5 largest contours

    #   find corner coordinates
    for c in ct:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenct = approx
            break

        # cv2.drawContours(image, approx, -1, (0, 255, 0), 2)
        # cv2.imshow('Outline', image)
        # cv2.waitKey(0)

    # #   show outline of receipt
    # cv2.drawContours(image, [screenct], -1, (0, 255, 0), 2)
    # cv2.imshow('Outline', image)
    # cv2.waitKey(0)

    #   apply transformation for a top down view
    try:
        transformed = transform_four_points(orig, screenct.reshape(4, 2) * ratio)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        success = True
    except(BaseException, Exception):
        print('** FAILED TO GET SCANNED VIEW, USING ORIGINAL IMAGE **')
        # transformed = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        transformed = orig
        success = False
    # if threshold:
        #   commenting this out coz layers size exceed limit
        # T = threshold_local(transformed, 11, offset=10, method='gaussian')
        # transformed = (transformed > T).astype('uint8') * 255
    if blur:
        transformed = cv2.medianBlur(transformed, 3)

    # #   show black and white top down view
    # cv2.imshow('final', imutils.resize(transformed, height=650))
    # cv2.waitKey(0)

    #   save scanned version
    cv2.imwrite(img, imutils.resize(transformed, height=650))
    return success


def get_small_bw(img):
    """
    take in an image and return smaller black and white version
    :param img: (string) image name in 'receipt' folder
    :return: save file to /tmp/ folder
    """
    image = cv2.imread(img)
    image = imutils.resize(image, height=720)  # 720 pixels height
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # rgb to grayscale
    # cv2.imshow('gray', gray)
    cv2.imwrite(img, gray)
    # cv2.waitKey(1)


# def ocr_api(img, output)    #, restaurant):
#     """
#     pass img to free ocr api and check if it is a receipt of said restaurant
#     :param img: (string) image path
#     :param output: (string) json file path to store output
#     :param restaurant: (string) restaurant name
#     :return: (string) result from api
#     """

#     #   free ocr url
#     url = 'https://api.ocr.space/parse/image'

#     #   Get APIkey
#     #   need to create a txt file named apikey.txt prior
#     #   apikey.txt contains your apikey 
#     #   (replace 'helloworld' with actual apikey)
#     #   should be in the same directory as this script
#     with open('apikey.txt') as f:
#         apikey = f.read()
#     f.close()

#     #   set up parameters
#     payload = {
#         'apikey': apikey,
#         'OCREngine': 2
#     }

#     #   pass image to ocr api
#     with open(img, 'rb') as f:
#         r = requests.post(url, files={receipt_file: f}, data=payload)
#     f.close()

#     #   dump response to json file
#     f = open(output, 'w')
#     json.dump(r.text, f)
#     f.close()

#     #   get parsed text
#     result = json.loads(r.text)
#     try:
#         text = result['ParsedResults'][0]['ParsedText']  # 0 for first image, diff if more files
#         print('Parsed from ocr api: \n' + text)
#         # print('"' + restaurant + '" found in receipt: ')
#         # print(restaurant in text)
#         return text
#     except(BaseException, Exception) as e:
#         print('Something went wrong!')
#         print('Error message from api: ')
#         print(result['ErrorMessage'][0])
        

def tess(img, legacy=False, lstm=False):
    """
    parse img with tesseract
    :param img: (string) image path
    :param legacy: (boolean) tesseract machine
    :param lstm: (boolean) tesseract machine
    :return: (string) result from tesseract
    """
    config = r'--oem 3'     # default
    if legacy and lstm:
        config = r'--oem 2'
    elif lstm:
        config = r'--oem 1'     # neural nets LSTM engine
    elif legacy:
        config = r'--oem 0'     # legacy engine
    im = cv2.imread(img)
    text = pytesseract.image_to_string(im, config=config)
    # print('Parsed from tesseract: \n'+text)
    # print('"' + restaurant + '" found in receipt: ')
    # print(restaurant in text)
    
    return text


def lambda_handler(event, context):
    
    s3 = boto3.resource('s3')
    
    #   Get bucket name
    bucket = event['Records'][0]['s3']['bucket']['name']
    
    #   Get file/key name
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    try:
        #   Download image as to tmp file
        tmp_filename='/tmp/receipt.jpg'
        s3.Bucket(bucket).download_file(key, tmp_filename)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist: s3://" + bucket + key)
        else:
            raise
    
    scan = False
    try:
        #   transform receipt into scanned version
        scan = get_scanned(tmp_filename, threshold=False, blur=True)
        suffix = key[key.rindex('.'):]
        key = key[:key.rindex('.')]
        scanned_filename = 'scanned/'+key[9:]+'-scanned'+suffix
        #   also upload scanned version to bucket
        s3.Bucket(bucket).upload_file(tmp_filename, scanned_filename)
    except (BaseException, Exception) as e:
        print(e)
        # get_small_bw(tmp_filename)
        # s3.Bucket(bucket).upload_file(tmp_filename, 'scanned/'+key[9:-5]+'-small.jpeg')
    
    
    #   pass through tesseract
    text = tess(tmp_filename, lstm=True)
    # print(text)
    
    full = {}
    full['BucketName'] = bucket
    full['FileName'] = key+suffix
    full['ParsedResult'] = text
    full['UsedOriginalPhoto'] = not scan
    
    print(json.dumps(full, indent=4, sort_keys=True))
    
    return {
        'statusCode': 200,
        'body': json.dumps(full)
    }

