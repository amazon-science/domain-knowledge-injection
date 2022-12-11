python3 ../experiments/finetune_on_downstream_task.py \
        --data_dir ../../business_knowledge_injection_data/dialogue_state_tracking_data \
        --model_type bart \
        --task state_tracking \
        --model_name facebook/bart-large \
        --output_dir ./runs \
        --active_domains train \
        --fact_format atomic_and_composite_facts \
        --adapter_size 768 \
        --adapter_num_heads 12 \
        --adapter_num_layers 2 \
        --adapter_list enc-12,dec-12 \
        --do_train \
        --do_eval \
        --evaluate_during_training \
        --per_gpu_train_batch_size 4 \
        --per_gpu_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 1000 \
        --max_seq_length 256 \
        --max_gen_length 256 \
        --learning_rate 1e-6 \
        --warmup_steps 0 \
        --save_steps 1000 \
        --eval_epochs 1 \
        --save_total_limit 20 \
        --patience 10 \
        --no_encoder_integration \
        --adapter_combo_method mean \
        --kprs_dev_path ./codebase/krs_benchmark/dev/single_domain_samples.json \
        --kprs_test_path ./codebase/krs_benchmark/test/single_domain_samples.json \
        --adapter_model_checkpoint ./models/train_atomic_and_composite_facts_from_scratch_maxlen-256_batch-64_lr-3e-05_warmup-0_epoch-1000_enc-12,dec-12/checkpoint-train_adapter-epoch_490-gs_325533
        # --adapter_model_checkpoint ./models/attraction_atomic_and_composite_facts_from_scratch_maxlen-256_batch-64_lr-3e-05_warmup-0_epoch-1000_enc-12,dec-12/checkpoint-attraction_adapter-epoch_320-gs_6099
        # --adapter_model_checkpoint ./models/hotel_atomic_and_composite_facts_from_scratch_maxlen-256_batch-64_lr-3e-05_warmup-0_epoch-1000_enc-12,dec-12/checkpoint-hotel_adapter-epoch_470-gs_4710
        # --adapter_model_checkpoint ./models/restaurant_atomic_and_composite_facts_from_scratch_maxlen-256_batch-64_lr-3e-05_warmup-0_epoch-1000_enc-12,dec-12/checkpoint-restaurant_adapter-epoch_210-gs_5486
