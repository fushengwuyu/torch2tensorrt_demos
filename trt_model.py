# author: sunshine
# datetime:2021/12/9 下午2:39

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTModelPredict:
    def __init__(self, engine_path, shape=(608, 608)):
        shape = (1, 3, shape[0], shape[1])
        self.engine = self.get_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.buffers = self.allocate_buffers(self.engine, 1)
        self.context.set_binding_shape(0, shape)

    def allocate_buffers(self, engine, batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:

            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dims = engine.get_binding_shape(binding)

            # in case batch dimension is -1 (dynamic)
            if dims[0] < 0:
                size *= -1

            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def get_engine(self, engine_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_path))
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def do_inference(self, img_in):

        inputs, outputs, bindings, stream = self.buffers
        inputs[0].host = img_in
        for i in range(2):
            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            self.context.execute_async(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]
