import evals
from evals.registry import Registry
from evals.api import CompletionFn
from evals.elsuite.modelgraded import ModelBasedEvaluator
from policy_rag.app import get_answer  # your RAG pipeline

class PolicyRAGCompletionFn(CompletionFn):
    def __call__(self, prompt: str, **kwargs):
        # Call your RAG system to produce an answer
        response = get_answer(prompt)
        return {"text": response}

def run_evaluation():
    registry = Registry()
    eval = registry.get_eval("policy_rag_eval")
    evaluator = ModelBasedEvaluator(
        completion_fn=PolicyRAGCompletionFn(),
        eval_spec=eval
    )
    results = evaluator.run()
    print("\n--- Policy RAG Evaluation Results ---")
    print(results)

if __name__ == "__main__":
    run_evaluation()
