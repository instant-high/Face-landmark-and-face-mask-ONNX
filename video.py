import cv2
import numpy as np
import argparse
from tqdm import tqdm
#from skimage import img_as_ubyte

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default="face.mp4")

args = parser.parse_args()

from kp_masker.kp_masker import KP_MASK
kp_mask = KP_MASK(model_path="kp_masker\98kp_masker.onnx", device="cuda")

# pre-aligned face
video = cv2.VideoCapture(args.input)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))       
fps = video.get(cv2.CAP_PROP_FPS)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

output = cv2.VideoWriter("result.mp4" ,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (512, 256))

background = np.zeros((256, 256, 3))#, dtype=np.uint8)
          		
for i in tqdm(range(length)):

    ret, aligned_face = video.read()
    aligned_face = cv2.resize(aligned_face, (256, 256))

    score, face_mask, keypoints = kp_mask.get_kp_mask(aligned_face)
    
    face_mask = cv2.GaussianBlur(face_mask,(15, 15), 0)
    
    for point in keypoints:
        x, y = float(point[0]), float(point[1])
        cv2.circle(aligned_face, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    stacked_image = np.concatenate((aligned_face, face_mask), axis=1)
    
    cv2.imshow("Result",stacked_image)
    output.write(stacked_image.astype(np.uint8))
    
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        output.release()
        break
		
cv2.destroyAllWindows()
output.release()
