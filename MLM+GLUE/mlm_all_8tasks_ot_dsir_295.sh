
models=( bert-base-uncased )
sizes=( 10000 20000 30000 40000 )
datasets=( cola mnli mrpc qnli qqp rte stsb sst2 )


ot_path="ot_2m_50k.pkl"
for model in "${!models[@]}"
do
    for dataset in "${!datasets[@]}"
    do
        for size in "${!sizes[@]}"
        do
        echo "'$model' => '${models[$model]}'"
        echo "'$dataset' => '${datasets[$dataset]}'"
        echo "'$size' => '${sizes[$size]}'"
        echo "data file path: ot_7/${datasets[$dataset]}_$ot_path"
        python run_mlm.py --model_name ${models[$model]} --dataset_name ot_7/${datasets[$dataset]}_$ot_path --data_size ${sizes[$size]} --task_name ${datasets[$dataset]} --method ot --batch_size 8 --max_length 294 --epochs 1 --learning_rate 1e-6
        done
    done
done