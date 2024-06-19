import onnx
import tf2onnx.convert

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA

H5_MODEL_PATH = "../../models/giiaa-hist_200k_base-inceptionresnetv2_loss-0.078.hdf5"
ONNX_MODEL_PATH = "../../models/giiaa_model.onnx"


def earth_movers_distance(y_true, y_predicted):
    return 0


nima = BaseModuleGIIAA(custom_weights=H5_MODEL_PATH)
nima.build()
nima.compile()

onnx_model, _ = tf2onnx.convert.from_keras(nima.nima_model)
onnx.save(onnx_model, ONNX_MODEL_PATH)
