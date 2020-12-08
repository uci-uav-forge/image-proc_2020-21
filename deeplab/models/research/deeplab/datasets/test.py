python3.7 ./build_new_pqr_data.py --image_folder="/Users/mandywoo/Documents/UAV-Forge/image-proc_2020-21/models/research/deeplab/datasets/PQR/dataset/JPEGImages" --semantic_segmentation_folder="/Users/mandywoo/Documents/UAV-Forge/image-proc_2020-21/models/research/deeplab/datasets/PQR/dataset/SegmentationClassRaw" --list_folder="/Users/mandywoo/Documents/UAV-Forge/image-proc_2020-21/models/research/deeplab/datasets/PQR/dataset/ImageSets" --image_format="jpg" --output_dir="/Users/mandywoo/Documents/UAV-Forge/image-proc_2020-21/models/research/deeplab/datasets/PQR/tfrecord"


python3.7 ./export_model.py \
  --logtostderr \
  --checkpoint_path="/Users/mandywoo/Documents/UAV-Forge/image-proc_2020-21/models/research/deeplab/datasets/PQR/exp/train_on_trainval_set/train/model.ckpt-5" \
  --export_path="/Users/mandywoo/Documents/UAV-Forge/image-proc_2020-21/models/research/deeplab/datasets/PQR/exp/train_on_trainval_set/export/frozen_inference_graph.pb" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=2 \
  --crop_size=448 \
  --crop_size=448 \
  --inference_scales=1.0

python deeplab/export_model.py --checkpoint_path=/code/models/research/deeplab/weights_input_level_17/model.ckpt-22000 --export_path=/code/models/research/deeplab/frozen_weights_level_17/frozen_inference_graph.pb --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --crop_size=2048 --crop_size=2048 --num_classes=3