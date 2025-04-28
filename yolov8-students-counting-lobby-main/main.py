import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        
#create a frame/window named RGB
#in which video "p" consisting the entry and exits of the people will open and run.
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('p.mp4')

#coco.txt contains object names
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

#initialize the count to 0
count=0
tracker=Tracker() #Tracker is a class file which updates and cleans the data

area1=[(494,289),(505, 499),(578,496),(530, 292)]
area2=[(548, 290),(600, 496),(637,493),(574,288)]

#store the id and co-ordinates(x,y) of entering people(going_in)
#and exiting people(going_out) in dictionary format as follows-- {id:(x,y),id(x,y)}
#initialize empty dicts
going_out={}
going_in={}

# store id's of people in list 
out_counter=[]
in_counter=[]

while True:    
    ret,frame = cap.read()  #read the contents of the video "p"
    if not ret:
        break


#    count += 1
#    if count % 3 != 0:
#        continue
    frame=cv2.resize(frame,(1020,500)) #set the frame size 
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    
    llist=[]
    for index,row in px.iterrows():
#        print(row)
 
        
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])

       # print("x1:",x1,"y1:",y1,"x2:",x2,"y2:",y2,"d:",d)
        
        c=class_list[d]
        if 'person' in c:
            llist.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(llist)
    for bbox in bbox_idx:
        x3,y3,x4,y4,idd=bbox
        result=cv2.pointPolygonTest(np.array(area2,np.int32),(x4,y4),False)
        if result>=0:
            going_out[idd]=(x4,y4)
            #print("going_out",going_out)
        if idd in going_out:
            result1=cv2.pointPolygonTest(np.array(area1,np.int32),(x4,y4),False)
            if result1>=0:
                cv2.circle(frame,(x4,y4),7,(255,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) #create a square frame
                                                                                                    #around the person
                                                                                                    # when he/she enters the
                                                                                                    #specified area

                cvzone.putTextRect(frame,f'{idd}',(x3,y3),1,1)
                if out_counter.count(idd)==0:
                    out_counter.append(idd)

        result2=cv2.pointPolygonTest(np.array(area1,np.int32),(x4,y4),False)
        if result2>=0:
            going_in[idd]=(x4,y4)
        if idd in going_in:
            result3=cv2.pointPolygonTest(np.array(area2,np.int32),(x4,y4),False)
            if result3>=0:
                cv2.circle(frame,(x4,y4),7,(255,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2) #create a square frame
                                                                                               #around the person
                                                                                               # when he/she enters the
                                                                                               #specified area
                cvzone.putTextRect(frame,f'{idd}',(x3,y3),1,1)
                if in_counter.count(idd)==0:
                    in_counter.append(idd)
                
                
    outc=len(out_counter) # count the number of exiting people
    inc=len(in_counter) # count the number of entering people

    try:
        with open("logs.txt","w") as l:
            l.write("People counting System\n")
            l.write("Number of people entered:"+str(inc)+"\nNumber of people exited:"+str(outc))
    except:
        print("Error storing the data in the file")
    finally:
        l.close()

        
    cvzone.putTextRect(frame,f'out:{outc}',(175,450),2,2) 
    cvzone.putTextRect(frame,f'in:{inc}',(300,450),2,2)
#    print("out_counter",out_counter,"\ncount:",len(out_counter)) #print the outgoing count
#    print("in_counter",in_counter,"\ncount:",len(in_counter)) #print the entering count
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(25,100,240),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(25,100,240),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

