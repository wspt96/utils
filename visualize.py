import os
import numpy as np
#import cv2
import matplotlib.pyplot as plt
import glob
import shutil as sh
root = '/home/mawenya/code/data/visdrone_fig/'
img = os.path.join(root,'images')
ann = os.path.join(root,'annotations')
dpnet = os.path.join(root,'dpnet')
res_path = '/home/mawenya/code/visdrone_fig/fig/'
def getrec(ann):
    result = np.loadtxt(ann,delimiter=',')
   # print(result)
    rec = result[:,:4]
  #  print(rec)
    return rec
def getcat(ann):
    ann_dirs = glob.glob(ann+'/*.txt')
    flag = np.zeros(11)
    for ann in ann_dirs:
        for i in range(11):
            if flag[i]>3:
               continue
            res = np.loadtxt(ann,delimiter=',',dtype=int)
            print(flag[i])
            if len(res.shape)==1:
               continue
            if(res[:,5]==int(i+1)).any() and (res[:,5]!=0).all():
              #sh.copy(ann.replace('annotations','images').replace('.txt','.jpg'),'./result/'+str(i)+'.jpg')
              img_dir = ann.replace('annotations','images').replace('.txt','.jpg')
              res_dir = ann.replace('annotations','dpnet')
              save_dir = './result/'+str(i)+'-'+str(int(flag[i]))+'.jpg'
              showrec(img_dir,ann,res_dir,save_dir)
              flag[i]+=1
              break  
def showrec(img_dir,ann_dir,res_dir,save_dir):
        recs = getrec(ann_dir)
        img = cv2.imread(img_dir)
        recs_res = getrec(res_dir)
        for rec in recs:
         #   print((float(rec[0]),float(rec[1])))
            if int(rec[2])*int(rec[3])<1000:
                           img = cv2.rectangle(img,(int(rec[0]),int(rec[1])),(int(rec[0]+rec[2]),int(rec[1]+rec[3])),(0,0,255),4)
            else:       
                           img = cv2.rectangle(img,(int(rec[0]),int(rec[1])),(int(rec[0]+rec[2]),int(rec[1]+rec[3])),(0,0,255),7)
        for rec in recs_res:
            img = cv2.rectangle(img,(int(rec[0]),int(rec[1])),(int(rec[0]+rec[2]),int(rec[1]+rec[3])),(0,255,0),2)
   #     cv2.imshow('vis',img)
        #cv2.waitKey(10000)
  #      print(img_dir)
        cv2.imwrite(save_dir,img)

def vis(img,ann):
    img_dirs = glob.glob(img+'/*.jpg')
    ann_dirs = glob.glob(ann+'/*.txt')
   # print(ann_dirs)
   # print(img_dirs)
    for i in range(len(img_dirs)):
 #/home/mawenya/code/data/visdrone_fig/images/9999936_00000_d_0000009.jpg               
        img_dir = img_dirs[i]
        prefix =  img_dir.split('/')[-1].split('.')[0]  
        ann_dir = os.path.join(ann,prefix+'.txt')
        result = os.path.join(dpnet,prefix+'.txt')
        '''
        for dirs in ann_dirs:
         #   print(img_dir.split('/')[-1].split('.')[0])
            if img_dir.split('/')[-1].split('.')[0] == dirs.split('/')[-1].split('.')[0]:
               ann_dir = dirs
               break
        '''
        print(img_dir,': ',ann_dir)
        if img_dir.split('/')[-1].split('.')[0] !=ann_dir.split('/')[-1].split('.')[0]:
           break
           print('Error: img dont match anntation file',img_dir)
        recs = getrec(ann_dir)
        recs_res = getrec(result)
        img = cv2.imread(img_dir)
        for rec in recs:
         #   print((float(rec[0]),float(rec[1])))
            img = cv2.rectangle(img,(int(rec[0]),int(rec[1])),(int(rec[0]+rec[2]),int(rec[1]+rec[3])),(0,0,255),8)
        for rec in recs_res:
            img = cv2.rectangle(img,(int(rec[0]),int(rec[1])),(int(rec[0]+rec[2]),int(rec[1]+rec[3])),(0,255,0),2)
        cv2.imshow('vis',img)
        #cv2.waitKey(10000)
        print(img_dir)
        cv2.imwrite('./result/'+prefix+'.jpg',img)
#    cv2.destroyAllWindows()
def combine(res_path):

    split = np.zeros([765,5,3])
    split[:,:,0] = 255
    split[:,:,1]=255
    split[:,:,2] = 255
    split2 = np.zeros([5,4090,3])
    split2[:,:,0] = 255
    split2[:,:,1]=255
    split2[:,:,2] = 255
    dirs = glob.glob(res_path+'*.jpg')
    i=0
   # print
    for dir in dirs:
       img =  cv2.imread(dir)
#       print(img.shape)
       img = cv2.resize(img,(1360,765),interpolation = cv2.INTER_CUBIC)
       #row = np.hstack([row,split,img])
       if i==0:
          row = img 
       else: 
          if i%3==0:
             if i==3:
                clo=row
             else:
                print(clo.shape,split2.shape,row.shape)
                clo = np.vstack([clo,split2,row])
            
             row=img
          else:
            print(row.shape,split.shape,img.shape)
            print(i,':',row.shape)
            row = np.hstack((row,split,img))
       i+=1
    clo = np.vstack([clo,split2,row])
    cv2.imwrite('./result.jpg',clo)

def matplot():
    ap_18 = [31.88,30.92,27.1,26.48,22.68,21.34,21.34,21.07,21.05,20.03]
    ar_18 = [90.63,50.48,40.57,38.94,38.59,35.41,37.97,35.58,36.41,33.27]
    name_18 = ['HAL-Retina-Net','DPNet','DE-FPN','CFE-SSEv2','RD4MS','L-H RCNN2','Faster R-CNN2','RefineDet+','DDFPN','YOLOv3-DP']
    ap_19 = [29.62,29.13,29.13,28.59,28.55,28.39,27.83,27.33,26.46,26.35]
    ar_19 = [42.37,46.05,44.53,42.72,44.02,43.34,46.81,45.23,38.42,42.28]
    name_19 = ['DPNet-ensemble','RRNet','ACM-OD','S+D','BetterFPN','HRDet','CN-DhVaSa','SGE-cascade R-CNN','EHR-RetinaNet','CNAnet']
    co_18= ['lightcoral', 'brown', 'red', 'red', 'salmon', 'darksalmon', 'sienna', 'mistyrose', 'tomato','coral']
    co_19= ['springgreen','mediumaquamarine','aquamarine','paleturquoise','cyan','deepskyblue','skyblue','dodgerblue','teal','palegreen']
    color_18='orangered'
    color_19='deepskyblue'
    plt.figure(figsize=(9,6.5),dpi=300)

    plt.xlabel('AP',fontsize=15)
    plt.ylabel('AR500',fontsize=15)
    plt.xticks(np.arange(20,32,2),fontsize=15)
    plt.yticks(np.arange(30,95,10),fontsize=15)
    plt.xlim((19,33))
    plt.ylim((32,95))
    markers = ['d','>','h','*','p', 'o', 'X', 's', 'H','^']
#    plt.figure(figsize=(5,6),dpi=60)
    for i in range(10):
        if i == 3:
             f19=plt.scatter(ap_19[i], ar_19[i], marker=markers[i],edgecolors='black', color=color_19, s=28, label=name_19[i])
             f18=plt.scatter(ap_18[i], ar_18[i], marker=markers[i],edgecolors='C8', color=color_18, s=31, label=name_18[i])
        else:
#        plt.plot(ap_18[i], ar_18[i], marker=markers[i], color=color_18[i], ms=20, label=name_18[i])
             if i>6 and i<9:
                    if i==8:
                       f18=plt.scatter(ap_18[i], ar_18[i], marker=markers[i], edgecolors='C8',color=color_18, s=32,label=name_18[i])
                    else:
                       f18=plt.scatter(ap_18[i], ar_18[i], marker=markers[i], edgecolors='C8',color=color_18, s=28,label=name_18[i])

                    f19=plt.scatter(ap_19[i], ar_19[i], marker=markers[i],edgecolors='black', color=color_19, s=31, label=name_19[i])
   #                 f18=plt.plot(ap_18[i], ar_18[i], marker=markers[i], color=color_18, ms=8, label=name_18[i])
             else:
                    f19=plt.scatter(ap_19[i], ar_19[i], marker=markers[i], edgecolors='black',color=color_19, s=31, label=name_19[i])
                    f18=plt.scatter(ap_18[i], ar_18[i], marker=markers[i], edgecolors='C8',color=color_18, s=31, label=name_18[i])
    #plt.subplot(212)
    plt.legend(bbox_to_anchor=(0., 1.12,1,0.05),columnspacing=160, loc='upper right',
           ncol=5,mode='expand', scatterpoints=1 , frameon=False, fontsize='medium')  
 # plt.annotate(model[i], (model_size[i] - horizontal[i], acc[i] + vertical[i]), fontsize=20) 
    plt.savefig(os.path.join('./', 'det18.png'))

    plt.close()          

matplot()
#getcat(ann)
#combine(res_path)   
#vis(img,ann)
