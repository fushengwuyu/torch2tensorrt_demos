# author: sunshine
# datetime:2021/12/9 下午3:43
import torch
from torchvision import models


def transform_to_onnx(model, shape, onnx_file_name, input_names=["input"], output_names=['boxes', 'confs']):
    batch_size = shape[0]
    dynamic = False
    if batch_size <= 0:
        dynamic = True
    if dynamic:
        x = torch.randn(shape, requires_grad=True)
        dynamic_axes = {name: {0: "batch_size"} for name in input_names + output_names}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes
                          )
        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn(shape, requires_grad=True)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')


if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    transform_to_onnx(model,
                      shape=(1, 3, 224, 224),
                      onnx_file_name="resnet50.onnx",
                      input_names=["input"],
                      output_names=["output"]
                      )
