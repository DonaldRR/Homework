from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imgPath', default='./quantizedImages/quantized_k_10_image5.jpg',
                    type=str, help='Image Path')
parser.add_argument('-k', '--nComponents', default=500., type=float, help='Number of Components')
parser.add_argument('-c', '--storeCompressedImg', action='store_true', help='Store Compressed Image')
parser.add_argument('-o', '--outputDir', default='./compressedImages/', type=str, help='Directory to output processed Image')
# parser.add_argument('-s', '--showImage', action='store_true', help='Show processed image')
args = parser.parse_args()

# Read Image
img = cv2.imread(args.imgPath)
# Compress Image
compressed_img, pca, shape = pcaCompress(img, args.nComponents)

# Save Image
img_dir, img_n = os.path.split(args.imgPath)
cv2.imwrite(os.path.join(args.outputDir, 'compressed_{}_{}'.format(args.nComponents,img_n)), pcaDecompress(compressed_img, pca, shape))

# Save Compressed Data
if args.storeCompressedImg:
    np.save(os.path.join(args.outputDir, 'compressed_{}_{}.npy'.format(os.path.splitext(img_n, args.nComponents)[1])),
            [compressed_img, pca, shape])