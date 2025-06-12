# src/evaluate.py

import os
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter # For majority voting
import mir_eval # Added: Explicitly import mir_eval for IDE resolution

# Project internal imports
from src.config import (
    ANNOTATIONS_DIR, SONG_FORM_LABELS, SR, HOP_LENGTH
)
from src.utils import (
    parse_salami_annotation, compute_mir_eval_metrics, frames_to_time, time_to_frames
)

def evaluate_model_on_dataset(model, data_loader, config, dataset_name="Test"):
    """
    주어진 데이터셋에 대해 모델의 송폼 인식 성능을 평가합니다.

    Args:
        model (torch.nn.Module): 학습된 모델.
        data_loader (torch.utils.data.DataLoader): 평가할 데이터 로더 (예: test_loader).
        config (module): config.py 모듈.
        dataset_name (str): 평가할 데이터셋의 이름 (로그 출력용).

    Returns:
        dict: 데이터셋의 평균 평가 지표.
    """
    model.eval() # Set model to evaluation mode
    
    all_predicted_segments = [] # List of (boundaries, labels) tuples for mir_eval
    all_ground_truth_segments = [] # List of (boundaries, labels) tuples for mir_eval

    print(f"Evaluating model on the {dataset_name} set...")
    with torch.no_grad(): # Disable gradient calculation
        pbar = tqdm(data_loader, desc=f"Evaluating {dataset_name} Set")
        for batch_idx, (features, boundaries, segments, lengths, padding_mask, data_ids) in enumerate(pbar):
            features = features.to(config.DEVICE)
            
            # Model prediction
            # lengths are passed for GRU's pack_padded_sequence
            boundary_preds_logits, segment_preds_logits = model(features, lengths)

            # Move predictions to CPU and convert to numpy arrays
            boundary_probs = torch.sigmoid(boundary_preds_logits).squeeze(-1).cpu().numpy() # (B, T_max)
            segment_probs = torch.softmax(segment_preds_logits, dim=-1).cpu().numpy() # (B, T_max, num_classes)
            
            for i in range(features.shape[0]): # Iterate over each sample in the batch
                original_length = lengths[i]
                data_id = data_ids[i]

                # --- 1. Extract Predicted Boundaries ---
                # Simple peak picking: probability > 0.5 and higher than neighbors
                pred_boundary_frames = []
                pred_boundary_frames.append(0) # Always include the start of the audio
                
                for j in range(1, original_length - 1):
                    if boundary_probs[i, j] > 0.5 and \
                       boundary_probs[i, j] > boundary_probs[i, j-1] and \
                       boundary_probs[i, j] > boundary_probs[i, j+1]:
                        pred_boundary_frames.append(j)
                
                # Always include the end of the audio (last frame)
                pred_boundary_frames.append(original_length - 1)
                
                # Remove duplicates and sort
                pred_boundary_frames = np.unique(pred_boundary_frames).astype(int)
                pred_boundaries_time = frames_to_time(pred_boundary_frames, config.HOP_LENGTH, config.SR)

                # --- 2. Extract Predicted Labels for Segments ---
                pred_segment_labels_for_mir_eval = []
                # Create intervals based on predicted boundaries
                # Line 74: mir_eval.util.boundaries_to_intervals usage
                pred_intervals = mir_eval.util.boundaries_to_intervals(pred_boundaries_time)

                for start_time, end_time in pred_intervals:
                    start_frame = time_to_frames(np.array([start_time]), config.HOP_LENGTH, config.SR)[0]
                    end_frame = time_to_frames(np.array([end_time]), config.HOP_LENGTH, config.SR)[0]
                    
                    # Ensure frame indices are within valid range and segment has at least 1 frame
                    start_frame = max(0, min(start_frame, original_length - 1))
                    end_frame = max(start_frame + 1, min(end_frame, original_length))

                    if start_frame >= end_frame:
                        pred_segment_labels_for_mir_eval.append('unknown') # Handle empty segments
                        continue

                    # Get frame-wise label predictions for the current segment
                    segment_frame_label_indices = np.argmax(segment_probs[i, start_frame:end_frame], axis=1)
                    
                    if len(segment_frame_label_indices) > 0:
                        # Majority voting for the segment label
                        most_common_label_idx = Counter(segment_frame_label_indices).most_common(1)[0][0]
                        pred_segment_labels_for_mir_eval.append(config.SONG_FORM_LABELS[most_common_label_idx])
                    else:
                        pred_segment_labels_for_mir_eval.append('unknown') # Should not happen if segment has frames

                # --- 3. Load Ground Truth Annotation ---
                gt_boundaries_time, gt_labels_str = parse_salami_annotation(
                    os.path.join(config.ANNOTATIONS_DIR, data_id, 'parsed', f'{data_id}.txt'),
                    config.SONG_FORM_LABELS
                )

                all_predicted_segments.append((pred_boundaries_time, pred_segment_labels_for_mir_eval))
                all_ground_truth_segments.append((gt_boundaries_time, gt_labels_str))

    # --- 4. Calculate Average mir_eval Metrics ---
    all_metrics_raw = {
        'detection_precision': [], 'detection_recall': [], 'detection_f_measure': [],
        'frame_based_accuracy': [], 'structure_f_measure': []
    }

    for i in range(len(all_predicted_segments)):
        metrics = compute_mir_eval_metrics(all_predicted_segments[i], all_ground_truth_segments[i])
        for key, value in metrics.items():
            all_metrics_raw[key].append(value)
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics_raw.items()}

    print(f"\n--- {dataset_name} Set Evaluation Results ---")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("-----------------------------------\n")

    return avg_metrics
