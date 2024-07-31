import cv2
import onnxruntime
import onnx
import numpy as np

'''
98kp_masker.onnx

Number of input nodes: 1
Number of output nodes: 3

Input Name: kp_input, Shape: [1, 3, 256, 256]

Output Name: kp_score, Shape: [1, 98]
Output Name: facemask, Shape: [1, 1, 256, 256]
Output Name: kp_output, Shape: [1, 196]
'''


class KP_MASK:
    def __init__(self, model_path="98kp_masker.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        model = onnx.load(model_path)

        
    def get_kp_mask(self, image):
    
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1)) / 255
        image = np.expand_dims(image, axis=0).astype(np.float32)
                
        score, mask, kp = self.session.run(None,{'kp_input':image})

        mask = mask.squeeze()
        mask = np.expand_dims(mask, axis=0)
        mask = mask.transpose((1, 2, 0))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask = mask.astype(np.uint8)
        mask = mask *  255

        kp = np.array(kp)[:98*2].reshape(-1,2)
        kp = kp * [256, 256]
        
        return score, mask, kp
