BATCH_SIZE=10
LEARNING_RATE=1e-5
EPOCHS=40

TRAIN_IMAGE_FOLDER=~/Documents/Machine\ Learning/datasets/COCO/train2017
TRAIN_ANNOTATION_FILE=~/Documents/Machine\ Learning/datasets/COCO/annotations/captions_train2017.json

TEST_IMAGE_FOLDER=~/Documents/Machine\ Learning/datasets/COCO/val2017
TEST_ANNOTATION_FILE=~/Documents/Machine\ Learning/datasets/COCO/annotations/captions_val2017.json

train_cnn_cnn:
	python train_cnn_cnn.py --batch_size 16 --image_folder $(TRAIN_IMAGE_FOLDER) --captions_file $(TRAIN_ANNOTATION_FILE) \
				--val_image_folder $(TEST_IMAGE_FOLDER) --val_captions_file $(TEST_ANNOTATION_FILE) \
				--batch_size $(BATCH_SIZE) --lr $(LEARNING_RATE) --epochs $(EPOCHS)

generate_mini_glove:
	python recortar_glove.py --file $(TRAIN_ANNOTATION_FILE)

inference:
	python inference.py --images $(TRAIN_IMAGE_FOLDER) --captions $(TRAIN_ANNOTATION_FILE)

dicts:
	python gen_dicts.py --file $(TRAIN_ANNOTATION_FILE)
