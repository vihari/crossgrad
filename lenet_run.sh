TRAIN_DIR=$HOME/sandbox/adaptive_lenet_logs

DATASET_DIR=/mnt/blossom/data/viharip/data/image_fonts/
DATASET_NAME=gfont
MODEL_NAME=crossgrad
TARGET_STYLE=False
mkdir $TRAIN_DIR
if true
then
    rm $TRAIN_DIR/*
    python train_image_classifier.py \
	   --train_dir=${TRAIN_DIR} \
	   --dataset_name=${DATASET_NAME} \
	   --dataset_split_name=train \
	   --dataset_dir=${DATASET_DIR} \
	   --model_name=${MODEL_NAME} \
	   --max_number_of_steps=50000 \
	   --learning_rate=0.02\
	   --save_interval_secs=100 \
	   --save_summaries_secs=100 \
	   --log_every_n_steps=2500 \
	   --batch_size=32 \
	   --target_style=${TARGET_STYLE} \
	   --eval_command="bash -v eval.sh" \
	   --eval_once_in=-1 \
    	   --ignore_missing_vars=True \
	   #--checkpoint_path=/mnt/a99/d0/viharip/sandbox/style_classifier_logs/ \
           #$HOME/sandbox/adaptive_lenet_logs4/ \
	   #--checkpoint_path=$HOME/sandbox/style_classifier_logs/
	   #1>$TRAIN_DIR/train_out 2>$TRAIN_DIR/train_err
fi

if true
then
   # Run evaluation.
   python eval_image_classifier.py \
	  --checkpoint_path=${TRAIN_DIR} \
	  --eval_dir=${TRAIN_DIR} \
	  --dataset_name=${DATASET_NAME} \
	  --dataset_split_name=train \
	  --target_style=${TARGET_STYLE} \
	  --dataset_dir=${DATASET_DIR} \
	  --model_name=${MODEL_NAME} \
	  --target_style=${TARGET_STYLE} \
	  --batch_size=1000 
   #1>$TRAIN_DIR/test_out 2>$TRAIN_DIR/test_err
fi

if true
then
    # Run evaluation.
    python eval_image_classifier.py \
	   --checkpoint_path=${TRAIN_DIR} \
	   --eval_dir=${TRAIN_DIR} \
	   --dataset_name=${DATASET_NAME} \
	   --target_style=${TARGET_STYLE} \
	   --dataset_split_name=validation \
	   --dataset_dir=${DATASET_DIR} \
	   --model_name=${MODEL_NAME} \
	   --target_style=${TARGET_STYLE} \
	   --batch_size=1000
    #1>$TRAIN_DIR/test_out 2>$TRAIN_DIR/test_err
fi

if true
then
    # Run evaluation.
    python eval_image_classifier.py \
	   --checkpoint_path=${TRAIN_DIR} \
	   --eval_dir=${TRAIN_DIR} \
	   --dataset_name=${DATASET_NAME} \
	   --target_style=${TARGET_STYLE} \
	   --dataset_split_name=test \
	   --dataset_dir=${DATASET_DIR} \
	   --model_name=${MODEL_NAME} \
	   --target_style=${TARGET_STYLE} \
	   --batch_size=1000 \
	   --metadir=${DATASET_DIR}/emb_meta
    #1>$TRAIN_DIR/test_out 2>$TRAIN_DIR/test_err
fi
