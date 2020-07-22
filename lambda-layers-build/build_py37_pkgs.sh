# Build Docker image containing Tesseract
docker build -t python_layer -f Dockerfile-py37 .

declare -a arr=("opencv-python" "pytesseract" "imutils")
## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   docker run --rm -v $(pwd):/package python_layer "$i"
done
