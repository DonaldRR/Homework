from tqdm import tqdm
from config import *
from utils import *


"""
numberOfClusters
initialSeedsFile
TweetsDataFile
outputFile
"""

import argparse

parser = argparse.ArgumentParser(description='Tweet K-means')
parser.add_argument('-n', '--numberOfClusters', default=25, type=int, help='Number of Clusters')
parser.add_argument('-i', '--initSeedsFile', default=INIT_URL, type=str, help='Initial Seeds File')
parser.add_argument('-d', '--TweetsDataFile', default=JSON_URL, type=str, help='Tweets Data File')
parser.add_argument('-o', '--outputFile', default=OUTPUT_PATH, type=str, help='Output File')
args = parser.parse_args()

# Read and Parse Json file from URL
print("# Read JSON ...")
jsons = parseJson(args.TweetsDataFile)

# Initialize Json file
jsons_ = {}
for t in jsons:
    id_ = t['id']
    text_ = t['text']
    jsons_[str(id_)] = {}
    jsons_[str(id_)]['text'] = text_
    jsons_[str(id_)]['cluster'] = 0
    jsons_[str(id_)]['distance'] = -1

# Read and Load initializatoin
print('# Initialize ...')
init_content = request.urlopen(args.initSeedsFile)
init_content = init_content.read().decode("utf-8").replace('\n', '').split(',')
clusters_means = [jsons_[init]['text'] for init in init_content][:args.numberOfClusters]
n_clusters = len(clusters_means)

# UPDATE_FLAG: boolean
#   True: Continue K-means
#   False: Stop algorithm
UPDATE_FLAG = True
# clusters: dictionary
#   Cluster are stored as lists indicated by cluster number, e.g. clusters['1']
clusters = {}
clusters_distribution = []
i = 0

print('# Training ...')
while UPDATE_FLAG:
    UPDATE_FLAG = False
    count_FLAG = 0
    print("## {}th iteration...".format(i + 1))

    # Updating each tweet
    for k, v in tqdm(jsons_.items()):
        c = v['cluster']
        d = v['distance']
        if d < 0:
            jsons_[k]['distance'] = JaccardDist(clusters_means[c], v['text'])
            d = jsons_[k]['distance']

        # Walk through means of all clusters
        for c_ in range(n_clusters):
            d_ = JaccardDist(clusters_means[c_], v['text'])

            if d_ < d:
                c = c_
                d = d_
                UPDATE_FLAG = True
                count_FLAG += 1

        # Assign updated cluster
        clusters_distribution.append(c)
        jsons_[k]['cluster'] = c
        jsons_[k]['distance'] = d

        try:
            clusters[str(c)].append(v['text'])
        except:
            clusters[str(c)] = []
            clusters[str(c)].append(v['text'])

    # Compute clusters' means
    for c in range(n_clusters):
        clusters_means[c] = computeMean(clusters[str(c)], 0.1)[0]

    print("## {} samples are updated in this round".format(count_FLAG))
    count_FLAG = 0

    clusters_distribution = []
    i += 1


# Form lines
clusters_txt = []
for i in range(n_clusters):
    line = '{}  '.format(i+1)
    ids = []
    for k, v in jsons_.items():
        if v['cluster'] == i:
            ids.append(k)
    ids = ','.join(ids)
    line += ids
    line += '\n'
    clusters_txt.append(line)

# Save
print('# Save ...')
out_file = open(args.outputFile,'w')
for i in range(n_clusters):
    out_file.write(clusters_txt[i])

print('# Finished.')