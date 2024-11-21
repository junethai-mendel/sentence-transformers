from __future__ import annotations

import heapq
import logging
import os
import random
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class PatientQAEvaluator(SentenceEvaluator):
    """
    This class evaluates an Patient QA setting.

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import PatientQAEvaluator

    # Load a model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load some examples
    corpus = [
        "Source: SOAP_Note. Date: 2023-05-11. Context: there has been modest improvement, but she continues to have severe arthritis. She is planning on talking to her rheumatologist about alternate therapies. ONCOLOGIC/HEMATOLOGIC HISTORY #Oncologic History: - {{06/2021}} Right foot wide local excision: Acral nodular malignant melanoma, Breslow thickness 2.7 mm with ulceration (pT3b). 10 mitoses/mm2, MSS: Negative, LVI: Negative, Perineural invasion: Negative, Margins: Negative. - 06/30/21 Right inguinal lymph node excision: 1/5",
        "Source: SOAP_Note. Date: 2023-11-16. Context: acral nodular malignant melanoma with a Breslow thickness of 2.7 mm with ulceration. There were 10 mitosis per mm2, no microsatellitosis, no LVI or PNI, and tumor cells extended to the peripheral tissue margin. She subsequently underwent excision of the melanoma with clear margins in 6/2021. She underwent right inguinal lymph node excision with 1/5 lymph nodes involved. She has completed adjuvant pembrolizumab as of 4/2023, discontinued just short of 1 year due to immunotherapy-induced arthralgias. Oncology history * {{06/2021}}",
        "Source: SOAP_Note. Date: 2023-01-09. Context: difficulty making a fist, difficulty gripping things, knee pain, and night sweat but denies any new skin lesions. Her arthritis is her sole complaint today. ONCOLOGIC/HEMATOLOGIC HISTORY #Oncologic History: - {{06/2021}} Right foot wide local excision: Acral nodular malignant melanoma, Breslow thickness 2.7 mm with ulceration (pT3b). 10 mitoses/mm2, MSS: Negative, LVI: Negative, Perineural invasion: Negative, Margins: Negative. - 06/30/21 Right inguinal lymph node excision: 1/5",
        "Source: SOAP_Note. Date: 2024-09-17. Context: acral nodular malignant melanoma with a Breslow thickness of 2.7 mm with ulceration. There were 10 mitosis per mm2, no microsatellitosis, no LVI or PNI, and tumor cells extended to the peripheral tissue margin. She subsequently underwent excision of the melanoma with clear margins in 6/2021. She underwent right inguinal lymph node excision with 1/5 lymph nodes involved. She has completed adjuvant pembrolizumab as of 4/2023, discontinued just short of 1 year due to immunotherapy-induced arthralgias. Oncologic history: * {{06/2021}}",
        "Source: SOAP_Note. Date: 2022-03-17. Context: with a Breslow thickness of 2.7 mm with ulceration. There were 10 mitosis per mm2, no microsatellitosis, no lymphovascular invasion, or perineural invasion, and tumor cells extended to the peripheral tissue margin. She subsequently underwent excision of the melanoma with clear margins in June 2021. He was then referred to Dr. ###### for sentinel lymph node biopsy. This showed 1 out of 4 lymph nodes were positive for metastatic melanoma without any evidence of distant metastasis. Given these findings, she was referred here for further evaluation and management. She initiated treatment with pembrolizumab on 9/3/2021. ONCOLOGIC/HEMATOLGIC HISTORY #Oncologic History: - {{06/2021}}",
        "Source: SOAP_Note. Date: 2021-12-22. Context: week for suture removal, who performed a biopsy that was benign. She last had a colonoscopy in July 2021 with Dr #####, which showed polyps but otherwise stable. ONCOLOGIC/HEMATOLGIC HISTORY #Oncologic History: - {{06/2021}} Right foot wide local excision: Acral nodular malignant melanoma, Breslow thickness 2.7 mm with ulceration (pT3b). 10 mitoses/mm2, MSS: Negative, LVI: Negative, Perineural invasion: Negative, Margins: Negative. - 06/30/21 Right inguinal lymph node excision: 1/5",
        "Source: SOAP_Note. Date: 2023-07-18. Context: to the PET CT findings. We will refer her to an ENT specialist per her request. She reports no issues with the left axillary biopsy she had on 06/23/23. ONCOLOGIC/HEMATOLOGIC HISTORY #Oncologic History: - {{06/2021}} Right foot wide local excision: Acral nodular malignant melanoma, Breslow thickness 2.7 mm with ulceration (pT3b). 10 mitoses/mm2, MSS: Negative, LVI: Negative, Perineural invasion: Negative, Margins: Negative. - 06/30/21 Right inguinal lymph node excision: 1/5",
        "Source: Radiology. Date: 2023-09-05. Context: PM Birth Date: 10/26/1983 Procedure Description: PET Wholebody Initial Gender: M PRESCRIPTION HISTORY: Melanoma of other part of the trunk. Melanoma staging. SUPPLEMENTAL HISTORY: Melanoma of the trunk treated with excision upper back and right lymph node removal on {{8/18/23}}. Appendectomy. COMPARISON: 8/17/2023. DOSE: 14.77 mCi fluorine-18 fluorodeoxyglucose. PROCEDURE: Following intravenous administration of F-18 FDG and a delay to allow for radiotracer uptake, a series of overlapping emission"
    ]
    corpus = {str(k): v for k,v in enumerate(corpus)}

    queries = [
        "when was the Wide Local Excision performed?",
        "when was the Sentinel Lymph Node Dissection performed?"
    ]
    queries = {str(k): v for k,v in enumerate(queries)}

    distractor_docs = {
        "0": ["0", "3"],
        "1": ["4", "5", "6"]
    }
    relevant_docs = {
        "0": set(["1", "2"]),
        "1": set(["7"])
    }
    qa_evaluator = PatientQAEvaluator(
        queries=queries,
        corpus=corpus,
        distractor_docs=distractor_docs,
        relevant_docs=relevant_docs,
        name="ontada2024-test"
    )
    results = qa_evaluator(model)
    """

    def __init__(
        self,
        queries: dict[str, str],  # qid => query
        corpus: dict[str, str],  # cid => doc
        relevant_docs: dict[str, set[str]],  # qid => Set[cid]
        distractor_docs: dict[str, list[str]],  # qid => List[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt: str | None = None,
        corpus_prompt_name: str | None = None,
    ) -> None:
        """
        Initializes the PatientQAEvaluator.

        Args:
            queries (Dict[str, str]): A dictionary mapping query IDs to queries.
            corpus (Dict[str, str]): A dictionary mapping document IDs to documents.
            relevant_docs (Dict[str, Set[str]]): A dictionary mapping query IDs to a set of relevant document IDs.
            corpus_chunk_size (int): The size of each chunk of the corpus. Defaults to 50000.
            mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
            ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
            accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
            precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
            map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
            show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
            batch_size (int): The batch size for evaluation. Defaults to 32.
            name (str): A name for the evaluation. Defaults to "".
            write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
            truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
            score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to the ``similarity`` function from the ``model``.
            main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
            query_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
            query_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.
            corpus_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
            corpus_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.
        """
        super().__init__()
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus = corpus
        # self.corpus_ids = list(corpus.keys())
        # self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.query_prompt = query_prompt
        self.query_prompt_name = query_prompt_name
        self.corpus_prompt = corpus_prompt
        self.corpus_prompt_name = corpus_prompt_name

        self.distractor_docs = distractor_docs
        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim

        if name:
            name = "_" + name

        self.csv_file: str = "Patient-QA_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self._append_csv_headers(self.score_function_names)

    def _append_csv_headers(self, score_function_names):
        for score_name in score_function_names:
            for k in self.accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in self.precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in self.mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in self.ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in self.map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, scores[name]["ndcg@k"][max(self.ndcg_at_k)]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        metrics = {
            f"{score_function}_{metric_name.replace('@k', '@' + str(k))}": value
            for score_function, values_dict in scores.items()
            for metric_name, values in values_dict.items()
            for k, value in values.items()
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    def compute_metrices(
        self, model: SentenceTransformer, corpus_model=None, corpus_embeddings: Tensor | None = None
    ) -> dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Compute embedding for the queries
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            query_embeddings = model.encode(
                self.queries,
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
            )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over the candidate chunks
        for qid, relevant_doc_ids in tqdm(
            self.relevant_docs.items(), desc="Iter Question", disable=not self.show_progress_bar
        ):
            distractor_doc_ids = self.distractor_docs[qid]
            candidate_doc_ids = relevant_doc_ids + distractor_doc_ids
            random.shuffle(candidate_doc_ids)

            candidate_docs = [self.corpus[cid] for cid in candidate_doc_ids]

            with (
                nullcontext()
                if self.truncate_dim is None
                else corpus_model.truncate_sentence_embeddings(self.truncate_dim)
            ):
                sub_corpus_embeddings = corpus_model.encode(
                    candidate_docs,
                    prompt_name=self.corpus_prompt_name,
                    prompt=self.corpus_prompt,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                )
            query_itr = self.queries_ids.index(qid)
            sub_query_embeddings = query_embeddings[query_itr].unsqueeze(0)
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(sub_query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for sub_corpus_id, score in zip(pair_scores_top_k_idx[0], pair_scores_top_k_values[0]):
                    corpus_id = candidate_doc_ids[sub_corpus_id]
                    if len(queries_result_list[name][query_itr]) < max_k:
                        heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))
                    else:
                        heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {"corpus_id": corpus_id, "score": score}

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores

    def compute_metrics(self, queries_result_list: list[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info("Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100))

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg
