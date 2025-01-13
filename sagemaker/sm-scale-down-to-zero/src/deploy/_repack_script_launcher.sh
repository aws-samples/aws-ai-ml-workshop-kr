
#!/bin/bash

var_dependencies="${SM_HP_DEPENDENCIES}"
var_inference_script="${SM_HP_INFERENCE_SCRIPT}"
var_model_archive="${SM_HP_MODEL_ARCHIVE}"
var_source_dir="${SM_HP_SOURCE_DIR}"

python _repack_model.py --dependencies "${var_dependencies}" --inference_script "${var_inference_script}" --model_archive "${var_model_archive}" --source_dir "${var_source_dir}"
