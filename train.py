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


from transformers import TrainerCallback


class CheckGradientNorm(TrainerCallback):
    def __init__(self, threshold):
        self.threshold = threshold

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        if total_norm > self.threshold:
            print(f"Gradient norm {total_norm} exceeds the threshold {self.threshold}")
            gradients = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()  # Calculate the L2 norm of gradients
                    gradients.append((name, grad_norm))
                else:
                    gradients.append((name, 0))  # No gradient for the parameter

            # Sort the list of tuples by gradient norm, from highest to lowest
            sorted_gradients = sorted(gradients, key=lambda x: x[1], reverse=True)

            # Print the top k gradients
            top_k = 3
            print(f"Top {top_k} parameter gradients:")
            for name, norm in sorted_gradients[:top_k]:
                print(f"{name}: {norm}")
            # import ipdb; ipdb.set_trace()
            # pass


@hydra.main(config_path=".", config_name="config.yaml")
def main(conf: DictConfig):
    # 1. Load a model to finetune with 2. (Optional) PEFT/LoRA
    model = SentenceTransformer(conf.model.name, trust_remote_code=True)
    model.max_seq_length = conf.model.max_seq_length
    model.tokenizer.padding_side = conf.model.tokenizer_padding_side
    model.set_pooling_include_prompt(conf.model.set_pooling_include_prompt)

    query_prompt = "Instruct: Given a question, retrieve passages that answer the question. Query: "
    prompts = {
        "question": query_prompt,
    }

    # 2. Apply PEFT with LoraConfig
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
    # train_dataset = dataset_dict["train"].select_columns(["question", "context", "event_type"])
    train_dataset = dataset_dict["train"].select_columns(["question", "context"])

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
        seed=conf.training.seed,
        prompts=prompts,
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
        query_prompt=query_prompt,
    )
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        # callbacks=[CheckGradientNorm(2)]
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
