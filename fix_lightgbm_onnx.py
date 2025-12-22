#!/usr/bin/env python3
"""
Fix LightGBM ONNX for MT5 Compatibility
- Removes ZipMap node (MT5 doesn't support map outputs)
- Sets fixed batch size of 1
- Ensures float32 outputs
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort
from pathlib import Path


def remove_zipmap_and_fix_batch(input_path: str, output_path: str) -> dict:
    """
    Remove ZipMap from LightGBM ONNX and fix batch dimension for MT5.
    
    Args:
        input_path: Path to input ONNX with ZipMap
        output_path: Path to save MT5-compatible ONNX
        
    Returns:
        dict with model info
    """
    print(f"Loading model from: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph
    
    # Find and analyze nodes
    zipmap_node = None
    classifier_node = None
    
    for node in graph.node:
        if node.op_type == "ZipMap":
            zipmap_node = node
        if node.op_type == "TreeEnsembleClassifier":
            classifier_node = node
    
    if not classifier_node:
        raise ValueError("No TreeEnsembleClassifier node found!")
    
    print(f"Found TreeEnsembleClassifier: {classifier_node.name}")
    print(f"ZipMap node: {zipmap_node.name if zipmap_node else 'Not found'}")
    
    # Get the probability output from classifier (before ZipMap)
    # TreeEnsembleClassifier outputs: [labels, probabilities]
    classifier_outputs = list(classifier_node.output)
    print(f"Classifier outputs: {classifier_outputs}")
    
    # Find the probability output (second output of classifier)
    prob_output_name = classifier_outputs[1] if len(classifier_outputs) > 1 else None
    
    if not prob_output_name:
        raise ValueError("Could not find probability output!")
    
    # Create new graph without ZipMap
    new_nodes = []
    cast_to_remove = set()
    identity_to_remove = set()
    
    for node in graph.node:
        if node.op_type == "ZipMap":
            continue  # Skip ZipMap
        if node.op_type == "Cast" and zipmap_node and node.output[0] in zipmap_node.input:
            # This Cast feeds into ZipMap, we'll keep it but redirect
            cast_to_remove.add(node.name)
            continue
        new_nodes.append(node)
    
    # Get feature count from classifier
    n_features = 27  # From config
    
    # Create new input with fixed batch size
    new_input = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        [1, n_features]  # Fixed batch=1 for MT5
    )
    
    # Create new outputs
    # 1. Label output (int64 -> we'll keep as is)
    label_output = helper.make_tensor_value_info(
        'label',
        TensorProto.INT64,
        [1]
    )
    
    # 2. Probability output (use the raw probabilities)
    # Need to find the actual probability tensor shape
    # TreeEnsembleClassifier outputs probabilities as [batch, n_classes]
    prob_output = helper.make_tensor_value_info(
        'probabilities', 
        TensorProto.FLOAT,
        [1, 3]  # batch=1, 3 classes (0=hold, 1=buy, 2=sell)
    )
    
    # Build new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name="LightGBM_MT5",
        inputs=[new_input],
        outputs=[label_output, prob_output],
        initializer=list(graph.initializer)
    )
    
    # Create new model
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    
    # Validate and save
    try:
        onnx.checker.check_model(new_model)
        print("Model validation passed!")
    except Exception as e:
        print(f"Warning: Model validation: {e}")
    
    onnx.save(new_model, output_path)
    print(f"Saved to: {output_path}")
    
    # Verify with inference
    print("\nVerifying inference...")
    sess = ort.InferenceSession(output_path)
    
    inp = sess.get_inputs()[0]
    print(f"Input: {inp.name}, shape: {inp.shape}")
    
    for out in sess.get_outputs():
        print(f"Output: {out.name}, shape: {out.shape}, type: {out.type}")
    
    # Test inference
    test_input = np.random.randn(1, n_features).astype(np.float32)
    outputs = sess.run(None, {'input': test_input})
    
    print(f"\nTest inference:")
    print(f"  Label: {outputs[0]}")
    print(f"  Probabilities: {outputs[1]}")
    
    return {
        "input_shape": list(inp.shape),
        "n_features": n_features,
        "n_classes": 3,
        "outputs": [out.name for out in sess.get_outputs()]
    }


def alternative_approach(input_path: str, output_path: str) -> dict:
    """
    Alternative: Create a simpler model that just uses the raw classifier outputs.
    """
    print(f"\nUsing alternative approach...")
    print(f"Loading: {input_path}")
    
    model = onnx.load(input_path)
    graph = model.graph
    
    # Find TreeEnsembleClassifier
    classifier = None
    for node in graph.node:
        if node.op_type == "TreeEnsembleClassifier":
            classifier = node
            break
    
    if not classifier:
        raise ValueError("No TreeEnsembleClassifier found!")
    
    # The classifier outputs are the raw outputs we need
    # Output 0: label (int64)
    # Output 1: probabilities (float32, [batch, n_classes])
    
    # Keep only essential nodes
    essential_nodes = [classifier]
    
    # Create inputs
    new_input = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        [1, 27]
    )
    
    # Create outputs matching classifier output names
    label_output = helper.make_tensor_value_info(
        classifier.output[0],
        TensorProto.INT64,
        [1]
    )
    
    prob_output = helper.make_tensor_value_info(
        classifier.output[1],
        TensorProto.FLOAT,
        [1, 3]
    )
    
    # Build minimal graph
    new_graph = helper.make_graph(
        nodes=essential_nodes,
        name="LightGBM_MT5_Minimal",
        inputs=[new_input],
        outputs=[label_output, prob_output],
        initializer=list(graph.initializer)
    )
    
    # Create model with correct opset for ML operators
    opset_imports = [
        helper.make_opsetid("", 15),  # ONNX opset
        helper.make_opsetid("ai.onnx.ml", 2)  # ML opset for TreeEnsembleClassifier
    ]
    
    new_model = helper.make_model(new_graph, opset_imports=opset_imports)
    new_model.ir_version = 8
    
    # Save
    onnx.save(new_model, output_path)
    print(f"Saved minimal model to: {output_path}")
    
    # Verify
    sess = ort.InferenceSession(output_path)
    inp = sess.get_inputs()[0]
    
    print(f"Input: {inp.name}, shape: {inp.shape}")
    for out in sess.get_outputs():
        print(f"Output: {out.name}, shape: {out.shape}")
    
    # Test
    test_input = np.random.randn(1, 27).astype(np.float32)
    outputs = sess.run(None, {inp.name: test_input})
    
    print(f"\nTest inference:")
    print(f"  Label: {outputs[0]}")  
    print(f"  Probabilities: {outputs[1]}")
    
    return {
        "input_shape": [1, 27],
        "n_features": 27,
        "n_classes": 3
    }


def main():
    input_path = "python_training/models/hybrid_lightgbm.onnx"
    output_path = "mt5_expert_advisor/Files/NeuralBot/hybrid_lightgbm.onnx"
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Try alternative approach (simpler and more reliable)
    try:
        info = alternative_approach(input_path, output_path)
        print(f"\n{'='*60}")
        print("SUCCESS: LightGBM ONNX fixed for MT5!")
        print(f"{'='*60}")
        print(f"Input shape: {info['input_shape']}")
        print(f"Features: {info['n_features']}")
        print(f"Classes: {info['n_classes']}")
        print(f"Output path: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
