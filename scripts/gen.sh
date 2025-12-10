device=0
model="path_to_opt1.3b"
num_samples=64
msig=4
data_name="c4"


median="hd" # online
python pmark.py --num_samples $num_samples --log_dir ${data_name}_logs/${model}_log/all-mpnet-base-v2_${median}${num_samples}_${msig} --model_path ../models/${model} --tokenizer_path ../models/${model} --pivot rand  --start 0 --end 100 --median_method ${median} --embedder_path ../models/all-mpnet-base-v2 --backend vllm --msig ${msig} --device0 ${device} --data_name ${data_name} --parallel 0 &
wait
median="prior" # offline
python pmark.py --num_samples $num_samples --log_dir ${data_name}_logs/${model}_log/all-mpnet-base-v2_${median}${num_samples}_${msig} --model_path ../models/${model} --tokenizer_path ../models/${model} --pivot rand  --start 0 --end 100 --median_method ${median} --embedder_path ../models/all-mpnet-base-v2 --backend vllm --msig ${msig} --device0 ${device} --data_name ${data_name} --parallel 0 &
wait
echo "Generation finished!"
# # # ##############################
median="hd" # online
CUDA_VISIBLE_DEVICES=${device} python detect.py --num_samples $num_samples --log_dir ${data_name}_logs/${model}_log --model_name ../models/${model}  --tokenizer_name ../models/${model} --sub_dir all-mpnet-base-v2_${median}${num_samples}_${msig} --pivot rand --start 0 --end 100 --median_method ${median}  --embedder_name ../models/all-mpnet-base-v2 --msig ${msig} --detect_origin True &

median="prior" # offline
CUDA_VISIBLE_DEVICES=${device} python detect.py --num_samples $num_samples --log_dir ${data_name}_logs/${model}_log --model_name ../models/${model}  --tokenizer_name ../models/${model} --sub_dir all-mpnet-base-v2_${median}${num_samples}_${msig} --pivot rand --start 0 --end 100 --median_method ${median}  --embedder_name ../models/all-mpnet-base-v2 --msig ${msig} --detect_origin True &

wait
echo "Detection finished!"
