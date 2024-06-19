
import onnx
import onnxruntime

ONNX_MODEL_PATH = "../../models/giiaa_model.onnx"

# model = onnx.load_model(ONNX_MODEL_PATH)

session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

