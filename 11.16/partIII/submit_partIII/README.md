# Color Quantization with K-means

## Folders and Files

    -- ProjectDir
    |
    |--- utils.py
    |--- color_quan.py
    |--- quantizedImages
       |--- xxx.jpg

## Dependencies and Environment

    * Python 3.6
    * sklearn
    * urllib
    * numpy
    * opencv-python

## Run

### Generate Quantized Image

```Bash
python color_quan.py [-d imgsDir[, -n imgName[, -k numberOfClusters[, -o outputFile]]]]
```

### Generate Compressed Image

```Bash
python img_compress.py [-i imgPath[, -k nComponents[, -o outputFile]]]
```