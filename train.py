from __future__ import annotations

import logging

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import PatientQAEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


@hydra.main(config_path=".", config_name="config.yaml")
def main(conf: DictConfig):
    # 1. Load a model to finetune with 2. (Optional) PEFT/LoRA
    model = SentenceTransformer(conf.model.name, trust_remote_code=True)
    model.max_seq_length = conf.model.max_seq_length
    model.tokenizer.padding_side = conf.model.tokenizer_padding_side
    model.set_pooling_include_prompt(conf.model.set_pooling_include_prompt)

    # Apply PEFT with LoraConfig
    embedding_model_peft_config = LoraConfig(
        target_modules=conf.peft.embedding_model.target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=conf.peft.inference_mode,
        r=conf.peft.r,
        lora_alpha=conf.peft.lora_alpha,
        lora_dropout=conf.peft.lora_dropout,
    )
    model[0].auto_model._modules["embedding_model"] = get_peft_model(
        model[0].auto_model._modules["embedding_model"], embedding_model_peft_config
    )

    latent_attention_model_peft_config = LoraConfig(
        target_modules=conf.peft.latent_attention_model.target_modules,
        inference_mode=conf.peft.inference_mode,
        r=conf.peft.r,
        lora_alpha=conf.peft.lora_alpha,
        lora_dropout=conf.peft.lora_dropout,
    )
    model[0].auto_model._modules["latent_attention_model"] = get_peft_model(
        model[0].auto_model._modules["latent_attention_model"], latent_attention_model_peft_config
    )

    # 3. Load a dataset to finetune on
    dataset_dict = load_from_disk(conf.data.hf_data_dir)
    train_dataset = dataset_dict["train"].select_columns(["id", "question", "context"])

    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # 5. Specify training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=conf.training.output_dir,
        num_train_epochs=conf.training.num_train_epochs,
        per_device_train_batch_size=conf.training.train_batch_size,
        per_device_eval_batch_size=conf.training.eval_batch_size,
        gradient_accumulation_steps=conf.training.gradient_accumulation_steps,
        learning_rate=conf.training.learning_rate,
        warmup_ratio=conf.training.warmup_ratio,
        fp16=conf.training.fp16,
        bf16=conf.training.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # important!
        eval_strategy=conf.training.eval_strategy,
        eval_steps=conf.training.eval_steps,
        save_strategy=conf.training.save_strategy,
        save_steps=conf.training.save_steps,
        save_total_limit=conf.training.save_total_limit,
        logging_steps=conf.training.logging_steps,
        logging_first_step=conf.training.logging_first_step,
        run_name=conf.training.run_name,
    )

    # 6. Create an evaluator & evaluate the base model
    eval_dataset = dataset_dict["test"]
    corpus_dataset = dataset_dict["corpus"]
    queries = dict(zip(eval_dataset["id"], eval_dataset["question"]))
    corpus = dict(zip(corpus_dataset["cid"], corpus_dataset["context"]))
    relevant_docs = dict(zip(eval_dataset["id"], eval_dataset["relevant_docs"]))
    distractor_docs = dict(zip(eval_dataset["id"], eval_dataset["distractor_docs"]))
    dev_evaluator = PatientQAEvaluator(
        corpus=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        distractor_docs=distractor_docs,
        show_progress_bar=True,
        name=conf.evaluation.name,
    )
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.remove_columns("id"),
        eval_dataset=eval_dataset.remove_columns(["id", "eval_id"]),
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # Evaluate the trained model on the evaluator after training
    dev_evaluator(model)

    # 8. Save the trained model
    model.save_pretrained(conf.output.model_save_path)

    # 9. Push it to the Hugging Face Hub
    model.push_to_hub(conf.output.run_name, private=conf.output.private)


if __name__ == "__main__":
    main()
