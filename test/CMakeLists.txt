#Testing macro
# `folder` may contain a subdirectory (e.g. ml-quant/aux); the source path uses it verbatim,
# while the executable target name sanitizes "/" -> "_" (CMake target names cannot contain slashes).
macro (add_test_executable_with_lib folder _name)
	string(REPLACE "/" "_" _tgt "test_${folder}_${_name}")
	add_executable(${_tgt} "${folder}/${_name}.cpp")
	target_link_libraries(${_tgt} ${EMP-ZK_LIBRARIES})
endmacro()

macro (add_test_case folder _name)
	add_test_executable_with_lib(${folder} ${_name})
	string(REPLACE "/" "_" _tgt "test_${folder}_${_name}")
  	add_test(NAME ${folder}_${_name} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_tgt}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_test_case_with_run folder _name)
	add_test_executable_with_lib(${folder} ${_name})
	string(REPLACE "/" "_" _tgt "test_${folder}_${_name}")
	add_test(NAME ${folder}_${_name} COMMAND "./run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_tgt}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")

endmacro()

# Standalone LogUpRangeCheck benchmark on a range-check workload. The executable
# runs its own setup_zk_arith / finalize_zk_arith cycle (cold cache + fresh VOLE
# pool) — driven across configurations by test/ram/bench_sweep.sh.
add_test_case_with_run(ram bench_logup)

# Standalone MSB / ReLU benchmark, comparable to CryptMatrix/ZKMath relu.cpp.
add_test_case_with_run(nonlinear bench_msb)

# zkGPT-style rounded division (two-sided range relation, no MSB).
add_test_case_with_run(nonlinear div)
add_test_case_with_run(nonlinear range_compare)
add_test_case_with_run(nonlinear rsqrt)

# Standalone inner-product matmul (reimplements emp-zk's QuickSilver IP proof,
# no singleton -> thread-able).
add_test_case_with_run(linear ip_matmul)
add_test_case_with_run(linear freivalds)
add_test_case_with_run(ml-fixed onnx_infer)
add_test_case_with_run(ml-fixed gpt2_prove)
add_test_case_with_run(ml-fixed bert_prove)
# ml-float (IEEE-754 float ResNet, Mystique-style) is DISABLED pending a port to the new emp
# Boolean-circuit API: Integer/Float/Bit moved to emp::rec, Bit lost .reveal(), Float_T's ctor
# changed. Re-enable after migrating volefier/ml-float/*.h. The ml-quant/ml-fixed paths are unaffected.
#add_test_case_with_run(ml-float float_nonlinear)
#add_test_case_with_run(ml-float float_convert)
#add_test_case_with_run(ml-float resnet)
#add_test_case_with_run(ml-float field_resnet)
#add_test_case_with_run(ml-float onnx_infer)
add_test_case(io safetensors)
add_test_case(io onnx_dump)
add_test_case(io onnx_fold)
add_test_case_with_run(ml-quant quant_onnx)
add_test_case_with_run(ml-quant quant_gpt2_prove)
add_test_case_with_run(ml-quant quant_qwen_prove)
add_test_case_with_run(ml-quant quant_bert_sym_prove)
add_test_case(io onnx_calib)
add_test_case(io precision_compare)
add_test_case_with_run(linear matmul_backward)
# add_test_case_with_run(ml-quant softmax_backward)
add_test_case_with_run(ml-fixed softmax_backward)
add_test_case_with_run(ml-fixed gelu_backward)
add_test_case_with_run(ml-fixed rmsnorm_backward)
add_test_case_with_run(ml-fixed attention_backward)
add_test_case_with_run(ml-fixed attention_backward_full)
add_test_case_with_run(ml-fixed ffn_backward)
add_test_case_with_run(ml-fixed block_train_step)
add_test_case_with_run(ml-fixed lora_train_step)
add_test_case_with_run(ml-fixed multiblock_lora_train_step)
add_test_case_with_run(ml-fixed lmhead_backward)
add_test_case_with_run(ml-fixed layernorm_backward)
add_test_case_with_run(ml-fixed sgd_optimizer)

add_test_case_with_run(ml-float float_convert_test)

add_test_case_with_run(ml-float onnx_infer)

add_test_case_with_run(ml-quant/aux bench_transformer)
add_test_case_with_run(ml-fixed/aux bench_transformer)
