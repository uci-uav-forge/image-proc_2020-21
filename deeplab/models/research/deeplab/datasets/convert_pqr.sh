CURRENT_DIR=$(pwd)
# WORK_DIR="./PQR"
WORK_DIR = "${CURRENT_DIR}/PQR"
PQR_ROOT="${WORK_DIR}/dataset"
SEG_FOLDER="${PQR_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${PQR_ROOT}/SegmentationClassRaw"

echo "Removing the color map in ground truth annotations..."
python3.7 remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

# IMAGE_FOLDER="${PQR_ROOT}/JPEGImages"
IMAGE_FOLDER="/Users/mandywoo/Documents/UAV-Forge/image-proc_2020-21/models/research/deeplab/datasets/PQR/dataset/JPEGImages"
LIST_FOLDER="${PQR_ROOT}/ImageSets"

echo ${IMAGE_FOLDER}

echo "Converting PQR dataset..."
python3.7 ./build_new_pqr_data.py \
# python3.7 ./build_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
