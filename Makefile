BATCH_SIZE=1
LEARNING_RATE=1e-5
EPOCHS=2000
N_LAYERS=20

TRAIN_IMAGE_FOLDER=~/Documents/Machine\ Learning/datasets/COCO/train2017
TRAIN_ANNOTATION_FILE=~/Documents/Machine\ Learning/datasets/COCO/annotations/captions_train2017.json

TEST_IMAGE_FOLDER=~/Documents/Machine\ Learning/datasets/COCO/val2017
TEST_ANNOTATION_FILE=~/Documents/Machine\ Learning/datasets/COCO/annotations/captions_val2017.json

train_cnn_cnn_ce:
	python train_cnn_cnn_ce.py --image_folder $(TRAIN_IMAGE_FOLDER) --captions_file $(TRAIN_ANNOTATION_FILE) \
				--val_image_folder $(TEST_IMAGE_FOLDER) --val_captions_file $(TEST_ANNOTATION_FILE) \
				--batch_size $(BATCH_SIZE) --lr $(LEARNING_RATE) --epochs $(EPOCHS) --n_layers $(N_LAYERS)

train_cnn_cnn_ha_ce:
	python train_cnn_cnn_ha_ce.py --image_folder $(TRAIN_IMAGE_FOLDER) --captions_file $(TRAIN_ANNOTATION_FILE) \
				--val_image_folder $(TEST_IMAGE_FOLDER) --val_captions_file $(TEST_ANNOTATION_FILE) \
				--batch_size $(BATCH_SIZE) --lr $(LEARNING_RATE) --epochs $(EPOCHS) --n_layers $(N_LAYERS)

generate_mini_glove:
	python recortar_glove.py --file $(TRAIN_ANNOTATION_FILE)

inference:
	python inference.py --images $(TEST_IMAGE_FOLDER) --captions $(TEST_ANNOTATION_FILE)
	#python inference.py --images $(TRAIN_IMAGE_FOLDER) --captions $(TRAIN_ANNOTATION_FILE)

dicts:
	python gen_dicts.py --file $(TRAIN_ANNOTATION_FILE)
