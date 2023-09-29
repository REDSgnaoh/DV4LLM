# cola
# mnli
# mrpc
# qqp
# rte
# sst2
# stsb
# wnli

# bert-base-uncased
# bert-base-cased
# bert-large-cased
# roberta-base
# roberta-large

# 3
# 4
# 5

# echo 


epochs=( 3 )
methods=( ot )
datasets=( cola )

runs=( 5000 6000 7000 8000 9000 10000 11000)
# runs=( 6000 7000 )


sizes=( 10000 20000 30000 40000 50000 )

model_path="bert_mlm_finetuned/bert-base-uncased/bs8/ep_1/lr_1e-06/max295/seed"

for run in "${!runs[@]}"
do
    for method in "${!methods[@]}"
        do

        for epoch in "${!epochs[@]}"
        do
            for dataset in "${!datasets[@]}"
            do

                for size in "${!sizes[@]}"
                do

                echo "'$epoch' => '${epochs[$epoch]}'"
                echo "'$method' => '${methods[$method]}'"
                echo "'$dataset' => '${datasets[$dataset]}'"
                echo "'$size' => '${sizes[$size]}'"
                echo "${runs[$run]}"
                echo "path: '$model_path/${methods[$method]}/cola/${sizes[$size]}'"
                python run_glue.py --model_name $model_path/${methods[$method]}/${datasets[$dataset]}/${sizes[$size]} --task_name ${datasets[$dataset]} --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --max_train_samples ${runs[$run]} --num_train_epochs ${epochs[$epoch]} --seed 0 --output_dir res/glue_select/${methods[$method]}/epoch${epochs[$epoch]}/${datasets[$dataset]}/${sizes[$size]}/${runs[$run]} --overwrite_output_dir

                done
            done
        done
    done
done
