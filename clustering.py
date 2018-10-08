
maps_js_key=r'AIzaSyALKN4Phkvtbfuls9KhSR0UVi7sbiJQvLk'



import   os                                     
import numpy as np                                              
import traceback

import matplotlib.pyplot as plt
from PIL import Image
from urllib.parse import urlsplit

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
import xml.etree.ElementTree as ET
import random
import cv2
import  re


import colorsys


from sklearn.metrics import silhouette_score

from maps_out import circles
###############################DBSCAN START


##with open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt') as f1:
#tg=open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt').read().split(";")



#matr=[]



#scores = open('C:\\Users\\victo\\Desktop\\data mining\\scores.txt').readlines()

##scores = open('C:\\Users\\victo\\Desktop\\data mining\\heidi.txt').readlines()


#streets=[]

#tags = {} 

#for s in scores:
#    s_=s.split('\n')[0].split(";") 
#    ################HEIDI
#    #del s_[0-1]
       
#    #new_list = ['1' if float(i) >0 else '0' for i in s_]
#    #matr.append(new_list)
#        ################HEIDI

#        ########MINE

#    if s_[1] not in tags:
#        tags[s_[1]]=dict.fromkeys(set(tg),0)
        
#        tags[s_[1]]['count_']=0
        
        
    
    
    
    
   
#    for i in range(2,len(s_)):
#        if float(s_[i])>0.75:

#            tags[s_[1]][tg[i-2]]+=float(s_[i])
#            tags[s_[1]]['count_']+=1
#        #if float(s_[i])>0.8:
#        #   # tags[tg[i-2]]+=1
#        #   l[i-2]=float(s_[i])

    
#for key in tags:

    
#    for key_ in tags[key]:


#        if key_!='count_':
#            tags[key][key_]=tags[key][key_]/tags[key]['count_']
#    del tags[key]['count_']     

#    streets.append(key)
#    matr.append(list(tags[key].values()))







#dbs=DBSCAN(eps=0.002, metric='euclidean')#, min_samples=50)

#pca = PCA(n_components=2)

#X=pca.fit(matr).transform(matr)





   




#print([list(tags[random.choice(list(tags.keys()))].keys())[i] for i in np.abs(pca.components_[0]).argsort()[::-1][:5]])
#print([list(tags[random.choice(list(tags.keys()))].keys())[i] for i in np.abs(pca.components_[1]).argsort()[::-1][:5]])


##nc=0 

##kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)


##kmeans = KMeans().fit(X)

#kmeans = dbs.fit(X)
#labels=kmeans.labels_.tolist()
#X=X.tolist()


#print(set(labels))
#print(len(set(labels)))





##with open('C:\\Users\\victo\\Desktop\\data mining\\clusters.txt','w') as file:

##    for i in range(len(streets)):
##        file.write(streets[i] +";"+str(labels[i])+'\n')



#colors = ["Red","Green","Blue","Magenta","Orange","Yellow","Purple","Azure","DarkGrey","AliceBlue","AntiqueWhite","Aqua","Aquamarine","Beige","Bisque","Black","BlanchedAlmond","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","HotPink","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Grey","GreenYellow","HoneyDew","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","OliveDrab","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","YellowGreen"]


#for i in range(len(labels)):
#    X[i].append(labels[i])

#plt.figure()

#for  i in range(len(set(labels))):
#   # plt.scatter(X[labels == i, 0], X[labels == i, 1], color=color, alpha=.8, lw=lw)


#    plt.scatter([row[0] for row in X if row[2]==i-1], [row[1] for row in X if row[2]==i-1], color=colors[i],alpha=.5, lw=2)
      
#plt.show()












#elements=''


#tree = ET.parse(r'C:\Users\victo\Desktop\data mining\allstreets.xml')
#root = tree.getroot()

#panos=open('C:\\Users\\victo\\Desktop\\data mining\\pano_ids.txt', encoding='utf-8-sig').readlines()

#for j in range(len(labels)):

#    label=labels[j]
#    street=streets[j]
    
#    elements=elements+('\''+street+'\''+r':{'+'\n'+r'coord:['+'\n')
    
#    for way in root.findall("way[@id='%s']"% (street,)):                                             #[@id='4483815'] waaggasse  [@id='4483815'] [@id='388474562']  v="Haldenstrasse"/>
#       # print(way.find("tag[@k='name']").get('v'))
#        wpts=[]
                  
        
#        for point in way.findall('nd'):
        
#            wpts.append(point.get('ref'))

#        for i in range(len(wpts)):
        
#            p=root.find("node[@id='"+wpts[i]+"']") 
         
#            lat=float(p.get('lat'))
#            lon=float(p.get('lon'))
         
#            elements=elements+(r'{lat:'+ p.get('lat')+r', lng:'+ p.get('lon')+r'}')
#            if i<(len(wpts)-1):
#                elements=elements+r','+'\n'
#            else:
#                elements=elements+'\n'+r'],'+'\n'+r'color:'+'\''+colors[label+1]+'\''+'\n'+r'}'
#    if j!=(len(labels)-1):
#        elements=elements+r','+'\n'
#    else:
#        elements=elements+'\n'













#f = open(r'C:\Users\victo\Desktop\data mining\plines_blank.html', "r")
#contents = f.readlines()
#f.close()

#contents.insert(37, elements)

#f = open(r'C:\Users\victo\Desktop\data mining\plines.html', "w")
#contents = "".join(contents)
#f.write(contents)
#f.close()






#street :{ 
#	coord:[
#          	{lat: 37.772, lng: -122.214},
#          	{lat: 21.291, lng: -157.821},
#          	{lat: -18.142, lng: 178.431},
#          	{lat: -27.467, lng: 153.027}
#        ],


#    color:'color'
#		},


################################DBSCAN END




################################ POINTS START
#################################################################################################


##with open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt') as f1:
#tg=open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt').read().split(";")



#matr=[]



#scores = open('C:\\Users\\victo\\Desktop\\data mining\\scores.txt').readlines()

##scores = open('C:\\Users\\victo\\Desktop\\data mining\\heidi.txt').readlines()


#streets=[]

#tags = {} 

#for s in scores:
#    s_=s.split('\n')[0].split(";") 
#    ################HEIDI
#    #del s_[0-1]
       
#    #new_list = ['1' if float(i) >0 else '0' for i in s_]
#    #matr.append(new_list)
#        ################HEIDI

#        ########MINE


        
    
  
    
    
#    l=[0]*len(tg)
#    for i in range(2,len(s_)):

#        if float(s_[i])>0.6:            #0.6 current optimal
#           # l[i-2]=1
#            l[i-2]=float(s_[i])
#    matr.append(l)





#dbs=DBSCAN(eps=0.35, metric='euclidean')#, min_samples=50)

#pca = PCA(n_components=2)

#X=pca.fit(matr).transform(matr)


#nc=8 

##kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)



#kmeans = dbs.fit(X)



##kmeans = KMeans().fit(X)

#labels=kmeans.labels_.tolist()
#X=X.tolist()


#print([tg[i] for i in np.abs(pca.components_[0]).argsort()[::-1][:5]])
#print([tg[i] for i in np.abs(pca.components_[1]).argsort()[::-1][:5]])





##with open('C:\\Users\\victo\\Desktop\\data mining\\clusters.txt','w') as file:

##    for i in range(len(streets)):
##        file.write(streets[i] +";"+str(labels[i])+'\n')



#colors = ["Red","Green","Blue","Magenta","Orange","Yellow","Purple","Azure","DarkGrey","AliceBlue","AntiqueWhite","Aqua","Aquamarine","Beige","Bisque","Black","BlanchedAlmond","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","HotPink","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Grey","GreenYellow","HoneyDew","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","OliveDrab","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","YellowGreen"]


#for i in range(len(labels)):
#    X[i].append(labels[i])

#plt.figure()

#for  i in range(len(set(labels))):
#   # plt.scatter(X[labels == i, 0], X[labels == i, 1], color=color, alpha=.8, lw=lw)


#    plt.scatter([row[0] for row in X if row[2]==i-1], [row[1] for row in X if row[2]==i-1], color=colors[i],alpha=.5, lw=2)
      
#plt.show()





#elements=''


#panos=open('C:\\Users\\victo\\Desktop\\data mining\\pano_ids.txt', encoding='utf-8-sig').readlines()

#fff = open(r'C:\Users\victo\Desktop\data mining\clusters.txt', "w")


#for i in range(len(labels)):

#    label=labels[i]
#    id_=scores[i].split('\n')[0].split(";")[0] 
#    fff.write(id_+r';'+str(label)+'\n')
#    for line in panos:
#        if id_ in line:

#            id,lat,lon,date_,head,street,st_name, node1,node2=line.split('\n')[0].split(";") 
#            if i!=(len(labels)-1):
                
#                elements=elements+('\''+id+'\''+r': {'+'\n'+r'center: {lat: '+lat+r', '+r'lng: '+lon+r'},'+'\n'+r'color:'+'\''+colors[label]+'\'\n'+r'},'+'\n'+'\n')
#            else:
#                elements=elements+('\''+id+'\''+r': {'+'\n'+r'center: {lat: '+lat+r', '+r'lng: '+lon+r'},'+'\n'+r'color:'+'\''+colors[label]+'\'\n'+r'}'+'\n'+'\n')


#fff.close()




#f = open(r'C:\Users\victo\Desktop\data mining\points_blank.html', "r")
#contents = f.readlines()
#f.close()

#contents.insert(37, elements)

#f = open(r'C:\Users\victo\Desktop\data mining\points.html', "w")
#contents = "".join(contents)
#f.write(contents)
#f.close()




################################POINTS END
#########################################################################################


##        chicago: {
##          center: {lat: 41.878, lng: -87.629},
##          population: 2714856
##          color:


##        },
##        newyork: {
##          center: {lat: 40.714, lng: -74.005},
##          population: 8405837
##        },
##        losangeles: {
##          center: {lat: 34.052, lng: -118.243},
##          population: 3857799
##        },
##        vancouver: {
##          center: {lat: 49.25, lng: -123.1},
##          population: 603502
##        }

##line 35




######################################HEIDI#######################################################
#####################################################################################################



##with open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt') as f1:
#tg=open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt').read().split(";")



#matr=[]



##scores = open('C:\\Users\\victo\\Desktop\\data mining\\scores.txt').readlines()

#scores = open('C:\\Users\\victo\\Desktop\\data mining\\Labels vs photos.csv').readlines()


#streets=[]

#tags = {} 


#tg=scores[0].split(";") [2:]

#del scores[:1]

#for s in scores:
#    s_=s.split('\n')[0].split(";") 
    
#    if s_[1]:#=='0':
#        del s_[:2]
       
#        new_list = [float(i) if float(i) >0.7 else 0.0 for i in s_]
#        matr.append(new_list)
    
       

     


        
    
  
    
    
#    #l=[0]*len(tg)
#    #for i in range(2,len(s_)):

#    #    if float(s_[i])>0.6:            #0.6 current optimal
#    #       # l[i-2]=1
#    #        l[i-2]=float(s_[i])
#    #matr.append(l)





#dbs=DBSCAN(eps=0.35, metric='euclidean')#, min_samples=50)

#pca = PCA(n_components=2)

#X=pca.fit(matr).transform(matr)


#nc=2 

#kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)



##kmeans = dbs.fit(X)



##kmeans = KMeans().fit(X)

#labels=kmeans.labels_.tolist()
#X=X.tolist()


#print(set([tg[i] for i in np.abs(pca.components_[0]).argsort()[::-1][:20]]+[tg[i] for i in np.abs(pca.components_[1]).argsort()[::-1][:20]]))
##print([tg[i] for i in np.abs(pca.components_[1]).argsort()[::-1][:20]])


#print(set(labels))


##with open('C:\\Users\\victo\\Desktop\\data mining\\clusters.txt','w') as file:

##    for i in range(len(streets)):
##        file.write(streets[i] +";"+str(labels[i])+'\n')



#colors = ["Red","Green","Blue","Magenta","Orange","Yellow","Purple","Azure","DarkGrey","AliceBlue","AntiqueWhite","Aqua","Aquamarine","Beige","Bisque","Black","BlanchedAlmond","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","HotPink","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Grey","GreenYellow","HoneyDew","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","OliveDrab","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","YellowGreen"]


#for i in range(len(labels)):
#    X[i].append(labels[i])

#plt.figure()

#for  i in range(len(set(labels))):
#   # plt.scatter(X[labels == i, 0], X[labels == i, 1], color=color, alpha=.8, lw=lw)


#    plt.scatter([row[0] for row in X if row[2]==i], [row[1] for row in X if row[2]==i], color=colors[i],alpha=.5, lw=2)
      
#plt.show()


#########################HEIDI END
##############################################


#####################################THE THING############################################################
############################################################################################################



##with open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt') as f1:


#scores = open('C:\\Users\\victo\\Desktop\\data mining\\Labels vs photos.csv').readlines()


#floor=0.7
#matr=[]
#matr_b=[]
#tg=scores[0].split(";") [2:]

#del scores[:1]

#for s in scores:
#    s_=s.split('\n')[0].split(";") 
    
#    if s_[1]:#=='0':
#        del s_[:2]
       
#        new_list = [float(i) if float(i) >floor else 0.0 for i in s_]
#        matr.append(new_list)
#    #if s_[1]=='1':
#    #    del s_[:2]
#    #    new_list = [float(i) if float(i) >floor else 0.0 for i in s_]
#    #    matr.append(new_list)
#    #    matr_b.append(new_list)
       

     


#tg_sv=open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt').read().split(";")



#matr_sv=[]

#pn=[]
#tags = {} 

#scores_sv = open('C:\\Users\\victo\\Desktop\\data mining\\scores.txt').readlines()       
    
  
#for s in scores_sv:
#    s_=s.split('\n')[0].split(";") 


    
#    tags[s_[0]]=dict.fromkeys(set(tg),0)
        
         
#    for i in range(2,len(s_)):
#        if float(s_[i])>floor and (tg_sv[i-2] in set(tg)):

#            tags[s_[0]][tg_sv[i-2]]=float(s_[i])
            
   
    
 


#for key in tags:

#    pn.append(key)
    

#    matr_sv.append(list(tags[key].values()))










##dbs=DBSCAN(eps=0.35, metric='euclidean')#, min_samples=50)

#pca = PCA(n_components=2)

#X=pca.fit(matr).transform(matr)


#Y=pca.transform(matr_sv)





#nc=2 

#kmeans = KMeans(n_clusters=nc).fit(X)


#labels_sv=kmeans.predict(Y).tolist()
##kmeans = dbs.fit(X)


##kmeans = KMeans().fit(X)

#labels=kmeans.labels_.tolist()
#X=X.tolist()


##print(set([tg[i] for i in np.abs(pca.components_[0]).argsort()[::-1][:20]]+[tg[i] for i in np.abs(pca.components_[1]).argsort()[::-1][:20]]))
##print([tg[i] for i in np.abs(pca.components_[1]).argsort()[::-1][:20]])


##print(set(labels))


#with open('C:\\Users\\victo\\Desktop\\data mining\\clusters_f.txt','w') as file:

#    for i in range(len(labels_sv)):
#        file.write(pn[i] +";"+str(labels_sv[i])+'\n')



#colors = ["Red","Green","Blue","Magenta","Orange","Yellow","Purple","Azure","DarkGrey","AliceBlue","AntiqueWhite","Aqua","Aquamarine","Beige","Bisque","Black","BlanchedAlmond","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","HotPink","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Grey","GreenYellow","HoneyDew","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","OliveDrab","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","YellowGreen"]


##for i in range(len(labels)):
##    X[i].append(labels[i])

##plt.figure()

##for  i in range(len(set(labels))):
##   # plt.scatter(X[labels == i, 0], X[labels == i, 1], color=color, alpha=.8, lw=lw)


##    plt.scatter([row[0] for row in X if row[2]==i], [row[1] for row in X if row[2]==i], color=colors[i],alpha=.5, lw=2)
      
##plt.show()


#hsl(120, 100%, 50%)






#elements=''


#panos=open('C:\\Users\\victo\\Desktop\\data mining\\pano_ids.txt', encoding='utf-8-sig').readlines()

#for i in range(len(labels_sv)):

#    label=labels_sv[i]
#    id_=pn[i]

#    for line in panos:
#        if id_ in line:

#            id,lat,lon,date_,head,street,st_name, node1,node2=line.split('\n')[0].split(";") 
#            if i!=(len(labels_sv)-1):
                
#                elements=elements+('\''+id+'\''+r': {'+'\n'+r'center: {lat: '+lat+r', '+r'lng: '+lon+r'},'+'\n'+r'color:'+'\''+colors[label]+'\'\n'+r'},'+'\n'+'\n')
#            else:
#                elements=elements+('\''+id+'\''+r': {'+'\n'+r'center: {lat: '+lat+r', '+r'lng: '+lon+r'},'+'\n'+r'color:'+'\''+colors[label]+'\'\n'+r'}'+'\n'+'\n')







#f = open(r'C:\Users\victo\Desktop\data mining\points_blank.html', "r")
#contents = f.readlines()
#f.close()

#contents.insert(37, elements)

#f = open(r'C:\Users\victo\Desktop\data mining\points_clustered.html', "w")
#contents = "".join(contents)
#f.write(contents)
#f.close()





##########################THE THING END###########################################
#####################################################################################




#####################################THE THING############################################################
############################################################################################################



##with open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt') as f1:


#scores = open('C:\\Users\\victo\\Desktop\\data mining\\Labels vs photos.csv').readlines()


#floor=0.7
#matr=[]
#matr_b=[]
#tg=scores[0].split(";") [2:]


     


#tg_sv=open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt').read().split(";")



#matr_sv=[]

#pn=[]
#tags = {} 

#scores_sv = open('C:\\Users\\victo\\Desktop\\data mining\\scores.txt').readlines()       
    



#for s in scores_sv:
#    s_=s.split('\n')[0].split(";") 


    

#    tags[s_[0]]=0 
         
#    for i in range(2,len(s_)):
#        if float(s_[i])>floor :

#            tags[s_[0]]+=1
            
   

#factor=1.0/sum(tags.values())

#tags = {k: v for k, v in tags.items() } 

#normalised = {k: (v-min(tags.values()))/(max(tags.values())-min(tags.values())) for k, v in tags.items() }    
 

#for key in tags:

#    pn.append(key)
    

#    matr_sv.append(normalised[key])












##with open('C:\\Users\\victo\\Desktop\\data mining\\clusters_f.txt','w') as file:

##    for i in range(len(labels_sv)):
##        file.write(pn[i] +";"+str(labels_sv[i])+'\n')



#colors = ["Red","Green","Blue","Magenta","Orange","Yellow","Purple","Azure","DarkGrey","AliceBlue","AntiqueWhite","Aqua","Aquamarine","Beige","Bisque","Black","BlanchedAlmond","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","HotPink","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Grey","GreenYellow","HoneyDew","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","OliveDrab","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","YellowGreen"]




#color_sv=[]


#for i in range(len(matr_sv)):

#    color_sv.append([120*matr_sv[i],100,100])


##regex = r'hsl\(\s*(\d+),\s*(\d+)%,\s*(\d+)%\s*\);'
##lines = [re.findall(regex,line) for line in color_sv.split('\n')]
#rgbs = [colorsys.hsv_to_rgb(line[0]/360,
#                            line[1]/100,
#                            line[2]/100) for line in color_sv]


#rgbhex = ["".join("%02X" % round(i*255) for i in rgb) for rgb in rgbs]




#elements=''


#panos=open('C:\\Users\\victo\\Desktop\\data mining\\pano_ids.txt', encoding='utf-8-sig').readlines()

#for i in range(len(matr_sv)):

  
#    id_=pn[i]

#    for line in panos:
#        if id_ in line:

#            id,lat,lon,date_,head,street,st_name, node1,node2=line.split('\n')[0].split(";") 
#            if i!=(len(matr_sv)-1):
                
#                elements=elements+('\''+id+'\''+r': {'+'\n'+r'center: {lat: '+lat+r', '+r'lng: '+lon+r'},'+'\n'+r'color:'+'\'#'+rgbhex[i]+'\'\n'+r'},'+'\n'+'\n')
#            else:
#                elements=elements+('\''+id+'\''+r': {'+'\n'+r'center: {lat: '+lat+r', '+r'lng: '+lon+r'},'+'\n'+r'color:'+'\'#'+rgbhex[i]+'\'\n'+r'}'+'\n'+'\n')







#f = open(r'C:\Users\victo\Desktop\data mining\points_blank.html', "r")
#contents = f.readlines()
#f.close()

#contents.insert(37, elements)

#f = open(r'C:\Users\victo\Desktop\data mining\points_number.html', "w")
#contents = "".join(contents)
#f.write(contents)
#f.close()





##########################THE THING END###########################################
#####################################################################################
class Label:
    
    def __init__(self,name,id,trainId,category,catId,hasInstances,ignoreInEval,color):
        self.name = name
        self.id = id
        self.trainId = trainId
        self.category = category
        self.catId = catId
        self.hasInstances = hasInstances
        self.ignoreInEval = ignoreInEval
        self.color = color




labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    #Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    #Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    #Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    #Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    #Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    #Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    #Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    #Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    #Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    #Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    #Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


 
##################### AVERAGE AMOUNT OF CATEGORIES




#with open(r"C:\\Users\\victo\\Desktop\\data elective\\images_segm_weimar\\labels_count.txt", "w") as out:

#    for input in os.listdir("C:\\Users\\victo\\Desktop\\data elective\\images_segm_weimar"):

#        if 'out' in input:
#            print (input)
#            values=np.zeros(len(labels))

#            input_image = cv2.cvtColor(cv2.imread("C:\\Users\\victo\\Desktop\\data elective\\images_segm_weimar\\"+input),cv2.COLOR_BGR2RGB)
    
#            for l in labels:
              

            

                
#               # print(input_image ==list(l.color))
                
#                values[l.trainId]=np.count_nonzero((input_image ==list(l.color)).all(axis = 2))/(input_image.size/3)
              
#                #values[l.trainId]=np.count_nonzero((input_image ==l.color).all())/input_image.size

#            print(sum(values))
#            ssss=input+";"+";".join([str(i) for i in values])+'\n'
#            print(input+";"+";".join([str(i) for i in values]))    
#            out.write(ssss)
###################### AVERAGE AMOUNT OF CATEGORIES





#####################################THE THING############################################################
############################################################################################################



#with open('C:\\Users\\victo\\Desktop\\data mining\\tags.txt') as f1:


city='zurich'
city_cap='Zurich'


#city='weimar'
#city_cap='Weimar'

#scores= open(("C:\\Users\\victo\\Desktop\\data elective\\images_segm_%s\\labels_count.txt" % (city))).readlines() 
scores=[]
#scores_in= open("C:\\Users\\victo\\Desktop\\data elective\\images_segm_zurich\\labels_count.txt" ).readlines() 

#scores_in.sort(key=lambda x: int(x.split('_')[0]))

#len1=len(scores_in)

#scores.extend(scores_in)

scores_in= open("C:\\Users\\victo\\Desktop\\data elective\\images_segm_weimar\\labels_count.txt" ).readlines() 

scores_in.sort(key=lambda x: int(x.split('_')[0]))

scores.extend(scores_in)



matr=[]
matr_b=[]
tg={l.trainId: l.name for l in labels}





for s in scores:
    s_=s.split('\n')[0].split(";")[1:]
    

       
    new_list = [float(i) for i in s_]
    matr.append(new_list)
   



pca = PCA(n_components=2)


#X=pca.fit(matr).transform(matr)
#X=X.tolist()


#print(set([tg[i] for i in np.abs(pca.components_[0]).argsort()[::-1][:10]]+[tg[i] for i in np.abs(pca.components_[1]).argsort()[::-1][:10]]))
#print([tg[i] for i in np.abs(pca.components_[0]).argsort()[::-1][:10]])
#print([tg[i] for i in np.abs(pca.components_[1]).argsort()[::-1][:10]])

epsilon=0

while True:
    

    X=pca.fit(matr).transform(matr)
    X=X.tolist()
    os.system('cls')
    epsilon=epsilon+0.001
    
    print(epsilon)


    dbs=DBSCAN(eps=0.073, metric='euclidean')#, min_samples=50)     #zrh 0.05  #weimar 0.09  #joint 0.073




    #Y=pca.transform(matr_sv)

   



    nc=2

    #kmeans = KMeans(n_clusters=nc).fit(matr)                   ####################KMEANS


    kmeans = dbs.fit(matr)                                      ####################DBSCAN


   

    #centers=kmeans.cluster_centers_.tolist()

    #print(centers)


    #labels_sv=kmeans.predict(Y).tolist()
    #kmeans = dbs.fit(X)


    #kmeans = KMeans().fit(X)
    klabels=kmeans.labels_.tolist()


    print(silhouette_score(X, klabels,metric='euclidean'))
    break


    if len([lbl for lbl in klabels if lbl>-1 ])>0:

        if len(set([lbl for lbl in klabels if lbl>-1 ]))==1:
            break
        else:

            #print(klabels)
            #X=X.tolist()

            #print(kmeans.inertia_)






            #print(set(labels))


            #with open("C:\\Users\\victo\\Desktop\\data elective\\images_segm_%s\\clusters_out.txt" % (city),'w')  as file:            ########NORMAL CLUSTER OUTPUT

            #    for i in range(len(klabels)):
            #        file.write(scores[i].split('\n')[0].split(";")[0] +";"+str(klabels[i])+'\n')




            with open("C:\\Users\\victo\\Desktop\\data elective\\joint_set\\clusters.txt" ,'w')  as file:                           ########JOINT CLUSTER OUTPUT

                for i in range(len(klabels)):
                    file.write(scores[i].split('\n')[0].split(";")[0] +";"+str(klabels[i])+'\n')


            #with open(r"C:\\Users\\victo\\Desktop\\data elective\\images_segm_weimar\\clusters_cen.txt",'w') as file:

            #    for i in range(len(centers)):
            #        for j in range(len(centers[i])):

            #            file.write(str(i)+";"+tg.get(j)+";"+str( centers[i][j])+'\n')



            colors = ["Red","Green","Blue","Magenta","Orange","Yellow","Purple","Azure","DarkGrey","AliceBlue","AntiqueWhite","Aqua","Aquamarine","Beige","Bisque","Black","BlanchedAlmond","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","HotPink","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Grey","GreenYellow","HoneyDew","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","OliveDrab","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","YellowGreen"]

            
            for i in range(len(klabels)):
                X[i].append(klabels[i])

            plt.figure()

            for  i in range(len(set(klabels))):
               # plt.scatter(X[labels == i, 0], X[labels == i, 1], color=color, alpha=.8, lw=lw)


                plt.scatter([row[0] for row in X if row[2]==i], [row[1] for row in X if row[2]==i], color=colors[i],alpha=.5, lw=2)
            plt.scatter([row[0] for row in X if row[2]==-1], [row[1] for row in X if row[2]==-1], color="Silver",alpha=.5, lw=2)

            ###########ADDITIONAL POINTS TO SEPARATE ZRH FROM WEIMAR




            plt.scatter([ x[0] for x in X[0:len1]],[ x[1] for x in X[0:len1]], marker="+" , color="Black" ,alpha=.5, lw=0.5)

            ###########ADDITIONAL POINTS TO SEPARATE ZRH FROM WEIMAR



            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
      
            plt.show()

            break
            #plt.savefig("C:\\Users\\victo\\Desktop\\data elective\\joint_set\\graphs\\%s.png" % ('%.3f' % epsilon))
            #plt.close('all')







    ##########################THE THING END###########################################
    ######################################################################################












#######MAP OUTPUT



#class point:
#    def __init__(self, latitude, longitude):
#        self.lat = latitude
#        self.lon = longitude

#maps_js_key=r'AIzaSyALKN4Phkvtbfuls9KhSR0UVi7sbiJQvLk'

#work_dir="C:\\Users\\victo\\Desktop\\data elective\\"
    




##city='zurich'
##city_cap='Zurich'


##city='weimar'
##city_cap='Weimar'
#for city,city_cap,klabels_slice, scores_slice in [['zurich','Zurich',klabels[:len1],scores[:len1]],['weimar','Weimar',klabels[len1:],scores[len1:]]]:

  

#    input1 = open((work_dir+"%s_360.csv") %(city_cap),'r') 

#    lines_pts=input1.readlines()[1:]
#    input1.close()


#    #input2 = open("C:\\Users\\victo\\Desktop\\data elective\\images_segm_%s\\clusters_out.txt" % (city),'r') 

#    #lines_clus=input2.readlines()[1:]
#    #input2.close()

#    #cir_prop = {label'rad': 5, 'strk_col': '#FF0000', 'stk_op': 0.8,'strk_w':2, 'fil_col': '#FF0000', 'fil_op':0.35}


#    pts=[]

#    for  i in range(len(klabels_slice)):
  
#       if klabels_slice[i]>=0:


#           label_col=colors[klabels_slice[i]]

#       else: 

#            label_col="Silver"

#           #col_hsv=[120*klabels[i],100,100]

#           #rgb = colorsys.hsv_to_rgb(col_hsv[0]/360,  col_hsv[1]/100, col_hsv[2]/100) 


#           #rgbhex = "".join("%02X" % round(i*255) for i in rgb) 

#       pts.append([point(float(lines_pts[i].split(',')[0]),float(lines_pts[i].split(',')[1])),   {'label':scores_slice[i].split('\n')[0].split(";")[0].split("_")[0] ,'rad': 5, 'strk_col': label_col, 'stk_op': 0.8,'strk_w':2, 'fil_col': label_col, 'fil_op':0.35}])

 




       
#       #    .split("_")[0]
   



#    out = open((work_dir+"joint_circles_%s.html")  % (city), "w") 

#    out.write(circles(pts, maps_js_key)   )
#    out.close()


#    #color_sv=[]    


#    #for i in range(len(matr_sv)):

#    #    color_sv.append([120*matr_sv[i],100,100])



#    #rgbs = [colorsys.hsv_to_rgb(line[0]/360,
#    #                            line[1]/100,
#    #                            line[2]/100) for line in color_sv]


#    #rgbhex = ["".join("%02X" % round(i*255) for i in rgb) for rgb in rgbs]






#    #####MAP OUTPUT