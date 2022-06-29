import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

import utils.inference as inference_utils # TRT/TF inference wrappers

import cv2

def main():
    engine_pth = '/dli/task/tensorrt_demos/mtcnn/det1.engine'
    trt_inference_wrapper = inference_utils.TRTInference(engine_pth, None,None,None,None)
    
    cap = cv2.VideoCapture(0)
    while True:
        ret,image_np = cap.read()
        import pdb;pdb.set_trace()
        
        break
    

if __name__=="__main__":
    main()

    
















# logger = trt.Logger(trt.Logger.WARNING)

# def prepare_engine(engine_pth):
#     """Deserialize a pre-made engine file"""
#     with trt.Runtime(logger) as runtime:
#         with open(engine_pth,'rb') as f:
#             serialized_engine = f.read()

        
#     engine = runtime.deserialize_cuda_engine(serialized_engine)
#     context = engine.create_execution_context()

#     # print engine info
#     inspector = engine.create_engine_inspector()
#     inspector.execution_context = context
#     print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

#     # create buffers
#     for binding in engine:
#         import pdb;pdb.set_trace()
#         size = trt.volume(engine.get_binding_shape(binding))*engine.max_batch_size
        
# if __name__ =="__main__":
#     engine_pth = '/dli/task/tensorrt_demos/mtcnn/det1.engine'
#     prepare_engine(engine_pth)
#     print('Awesome')
# # create a builder
# # builder = trt.Builder(logger)

# # import pdb;pdb.set_trace()
# # # create a network def in python
# # network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# # parser = trt.OnnxParser(network, logger)

# # model_pth = 'test.onnx'
# # success = parser.parse_from_file(model_path)
# # for idx in range(parser.num_errors):
# #     print(parser.get_error(idx))

# # if not success:
# #     pass # Error handling code here

# # deserialize engine
# det1_engine_pth = '/dli/task/tensorrt_demos/mtcnn/det1.engine'
# # with open(det1_engine_pth,"rb") as f:
# #     serialized_engine=f.read()

# # runtime = trt.Runtime(logger)
# # engine = runtime.deserialize_cuda_engine(serialized_engine)

# # # perform inference
# # context = engine.create_execution_context()

# # # print engine info
# # inspector = engine.create_engine_inspector()
# # inspector.execution_context = context
# # print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

# # import pdb;pdb.set_trace()
# # input_idx = engine[input_name]
# # output_idx = engine[output_name]

# # buffers = [None] * 2 # Assuming 1 input and 1 output
# # buffers[input_idx] = input_ptr
# # buffers[output_idx] = output_ptr

# # context.execute_async_v2(buffers, stream_ptr)
# # # import pdb;pdb.set_trace()
# # print('Awesome')