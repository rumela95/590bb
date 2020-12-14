import requests
import urllib2
import os
import re
from threading import *

def getAll(folder_name,html):
    d_files = {}
    for x in html:
        if " " in x:
            d_files[x.split(" ")[0]]=x.split(" ")[1]
            # print("\nvalue:" + d_files[x.split(" ")[0]])
    count = 0
    for name, link in d_files.items():
    # link = d_files["n12513613_1"]
        if name in image_list:
            img_valid.write(name + "\n")
            try:
                fname = "Dataset/" + folder_name + "/" + name + "_" + link.split("/")[-1][:-1]
                fname = re.sub(".jpg(\w*\W*)*",".jpg",fname)
                r = requests.get(link, allow_redirects=True, stream=True)
                # print(os.getcwd())
                if r.status_code == 200:
                    x = open(fname, "w+")
                    for chunk in r.iter_content(chunk_size=1024): 
                        if chunk: # filter out keep-alive new chunks
                            x.write(chunk)
                    # x.write(r.content)
                    count = count + 1
            except:
            #     print(e)
                pass
    pname = "Dataset/" + folder_name + "/" + "image_count.txt"
    p = open(pname, "w+").write("Total Count:" + str(count))


f = open("entity2id.txt","r")
parent_path = "/Users/rumelaghosh/cs590bb/data/Dataset/"
g1 = open("image_list.txt", "r")
image_list = []
img_valid = open("image_list_valid.txt", "a+")
for k in g1:
    image_list.append(k[:-2])

for line in f:
    try:
        folder_name = line.split("\t")[0]
        url = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=" + folder_name
        response = urllib2.urlopen(url)
        html = response.read().split("\n")
        cur_path = parent_path+folder_name
        os.mkdir(cur_path)
        t = Thread(target = getAll, args = (folder_name,html,))
        t.start()
    except:
        pass
    



# r = requests.get(url, allow_redirects=True)