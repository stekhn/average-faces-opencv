# Average faces with OpenCV (Python)

Calculate an average face from multiple images using the machine learning library **dlib** and the computer vision toolkit **OpenCV**. For this example, we'll use images of Bavarian politicians. These instructions and the some of the code build on Satya Mallick's excellent introduction to computer vision: [Learn OpenCV](http://www.learnopencv.com/average-face-opencv-c-python-tutorial/).

![Average face example](example.jpg)

## Data

I put together a list of all members of the Bavarian parliament, which I scraped from their [website](https://www.bayern.landtag.de/politicians/politicians-von-a-z/). The dataset contains unique IDs which we'll use to download the image of each politician.

Here the first five entries from `data/politicians.csv`:

```
id,forename,name,title,party,gender
555500000394,Klaus,Adelt,,SPD,M
555500002811,Ilse,Aigner,,CSU,W
555500000369,Hubert,Aiwanger,,Freie Wähler,M
555500000366,Horst,Arnold,,SPD,M
555500000341,Inge,Aures,Dipl.-Ingenieurin (FH); M.A.,SPD,W
```

## Requirements

[Python 2](https://www.python.org/downloads/) is required for running the scripts, though it might work with Python 3 as well. [OpenCV Python](https://pypi.python.org/pypi/opencv-python) uses pre-compiled OpenCV binary and can be installed using the Python package manager [pip](https://pypi.python.org/pypi/pip). Make sure to remove previous or other versions of OpenCV, to avoid conflicts. 

[Dlib](http://dlib.net/), which we'll use for landmark extraction, requires CMake to build:

```
$ brew install cmake
```

I've tested the scripts on a Mac running High Sierra (10.13). Linux users might need to change a few commands (like `apt-get install`) to set up their system and get the code to run. To avoid Python dependency trouble, we'll use the Python virtual environment wrapper [virtualenv](https://virtualenv.pypa.io/en/stable/).

## Setup

Update your Python package manager:

```
pip install --upgrade pip
```

Create a new virtual environment:

```
$ virtualenv venv
```

Activate the virtual environment:

```
$ source venv/bin/activate
```

Check if the Python virtual environment is set up correctly:

```
$ which python
/Users/your-username/Development/venv/env/bin/python
```

Install dependencies:

```
$ pip install -r requirements.txt
```

## Download images

*If you already have your own set of images, you can skip this step.*

Run the script proving a path to where the images should be downloaded. If the folder does not exist, it will be created for your convenience:

```
$ python download.py ./images
```

To download the images needed for averaging, the script stitches together an image URL using the base URL and the ID of each politician from the CSV file `data/politicians.csv`.

```
"https://www.bayern.landtag.de/images/politicians/" + "555500000394" + ".jpg"
```

## Extract face landmarks

The script tries to find human faces in an image and extract 68 landmarks. These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth. We'll need this landmarks to map the different faces onto each other.

The script needs a pre-trained model for predicting these features, which is available for download (~ 60 MB):

```
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

Unzip the shape predictor (~ 95 MB):

```
$ bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
```

```
$ python extract.py shape_predictor_68_face_landmarks.dat ./images
```

The extracted landmarks will be saved as list of xy coordinates in the same folder as the images, using a ".txt" extension.

See the references for more info about facial landmark recognition.

## Average faces

Bring the images to the same size and roughly align the images using the position of the eyes. Other features of the face might be misaligned. Therefore, we'll use a bounding box to triangulate the landmark points ([Delaunay Triangulation](http://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/)). These triangles can then be warped to match the other triangles, so the faces line up neatly. Finally the images will be blended together by applying some transparency.

To run the script, provide the path to your image folder. The folder should contain both images (.jpg) and landmarks (.txt). Optionally, you can specify the desired output size for the output image (width, height):

```
$ python average.py ./images 170 240
```

If need a detailed explanation on how this works, head over to [Learn OpenCV](http://www.learnopencv.com/average-face-opencv-c-python-tutorial/).

## References

Papers on which feature extraction methods used in dlib are based:

- C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. [300 faces In-the-wild challenge: Database and results](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_2016_imavis.pdf). Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
- C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. [A semi-automatic methodology for facial landmark annotation](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_cvpr_2013_amfg_w.pdf). Proceedings of IEEE Int’l Conf. Computer Vision and Pattern Recognition (CVPR-W), 5th Workshop on Analysis and Modeling of Faces and Gestures (AMFG 2013). Oregon, USA, June 2013.
- C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. [300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_iccv_2013_300_w.pdf). Proceedings of IEEE Int’l Conf. on Computer Vision (ICCV-W), 300 Faces in-the-Wild Challenge (300-W). Sydney, Australia, December 2013.
