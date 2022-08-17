import numpy as np
import cv2

video_path ='C:/Users/anu07/Desktop/GitHub/DeepLearning_OpenCV/images/shore.mov'
capture = cv2.VideoCapture(video_path)

all_rows = open(r'C:/Users/anu07/Desktop/GitHub/DeepLearning_OpenCV/model/synset_words.txt').read().strip().split('\n')

classes = [ r[r.find(' ')+ 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('C:/Users/anu07/Desktop/GitHub/DeepLearning_OpenCV/model/bvlc_googlenet.prototxt',
                                'C:/Users/anu07/Desktop/GitHub/DeepLearning_OpenCV/model/bvlc_googlenet.caffemodel')


if capture.isOpened() == False:
    print('Cannot open the file')

while True:
    ret, frame = capture.read()

    blob = cv2.dnn.blobFromImage(frame, 1, (224,224))
    net.setInput(blob)
    outp = net.forward()
    r = 1
    for i in np.argsort(outp[0])[::-1][:5]:
        txt = ' "%s" Probablity "%.3f" ' % (classes[i], outp[0][i]*100)
        cv2.putText(frame, txt, (0, 25 + 40*r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        r +=1
    
    if ret == True:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == 27:   #wait for 25 msec and wait for escape key
             break
    else:
        break

capture.release()
capture.destroyAllWindows()
