from utils import *
import argparse

"""
Color Quantization with K-means
"""

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--imgsDir', default='http://www.utdallas.edu/~axn112530/cs6375/unsupervised/images/',
                    type=str, help='Images Directory')
parser.add_argument('-n', '--imgName', default='image1.jpg', type=str, help='Image Name')
parser.add_argument('-k', '--numberOfClusters', default=3, type=int, help='Number of Clustering Colors')
parser.add_argument('-o', '--outputDir', default='./', type=str, help='Directory to output processed Image')
# parser.add_argument('-s', '--showImage', action='store_true', help='Show processed image')
args = parser.parse_args()

print('# Loading image {} ...'.format(args.imgName))
# Read and Parse image from url
url = path.join(args.imgsDir, args.imgName)
img = parseUrl2Img(url)
print('# Color Quantization processing ...')
# Color Quantization
img = colorQuan(img, args.numberOfClusters)

print('# Save image ...')
# Save image
cv2.imwrite(path.join(args.outputDir, 'processed_'+args.imgName), img)

# if args.showImage:
#     plt.figure(0)
#     plt.imshow(img)

print('# Finished.')