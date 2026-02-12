import json
import os
from typing import Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag import SimpleRAG, format_context

MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SYSTEM_PROMPT = "You are a helpful assistant."
DMN_MAX_NEW_TOKENS = int(os.environ.get("DMN_MAX_NEW_TOKENS", "400"))
DMN_MAX_RETRIES = int(os.environ.get("DMN_MAX_RETRIES", "1"))
POLICY_MAX_NEW_TOKENS = int(os.environ.get("POLICY_MAX_NEW_TOKENS", "180"))
DEMO_MODE = os.environ.get("DEMO_MODE", "full").lower()
DMN_SCHEMA_HINT = (
    "Use schema: decisionModel {id, name, inputs[{id,type}], output{id,type}, rules[{id,if,then}]}. "
    "Output JSON only."
)

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_base(device):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return tok, model

@torch.no_grad()
def generate(tok, model, prompt, max_new_tokens=220, temperature=0.2):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)

def apply_chat_template(tok, messages, add_generation_prompt=False):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    text = ""
    for msg in messages:
        if msg["role"] == "system":
            text += f"<|system|>\n{msg['content']}\n"
        elif msg["role"] == "user":
            text += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            text += f"<|assistant|>\n{msg['content']}\n"
    if add_generation_prompt:
        text += "<|assistant|>\n"
    return text

def build_messages(user_question, context=None, force_dmn=False):
    sys = SYSTEM_PROMPT
    if force_dmn:
        sys += " Output ONLY DMN JSON (no explanations)."

    if context:
        if force_dmn:
            user = (
                "Use the context sources if helpful. Output JSON only.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{user_question}"
            )
        else:
            user = (
                "Use the context sources to answer. Cite sources by id.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{user_question}"
            )
    else:
        user = user_question

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

def make_prompt(tok, user_question, context=None, force_dmn=False):
    messages = build_messages(user_question, context=context, force_dmn=force_dmn)
    return apply_chat_template(tok, messages, add_generation_prompt=True)

def extract_json(text: str) -> Tuple[Any, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no_json_object"
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate), ""
    except Exception as exc:
        return None, f"invalid_json: {exc}"

def validate_dmn(obj: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return ["top_level_not_object"]
    extra_top = set(obj.keys()) - {"decisionModel"}
    if extra_top:
        errors.append(f"top_level_extra_keys:{sorted(extra_top)}")
    dm = obj.get("decisionModel")
    if not isinstance(dm, dict):
        return ["missing_decisionModel_object"]

    allowed_dm_keys = {"id", "name", "inputs", "output", "rules"}
    extra_dm = set(dm.keys()) - allowed_dm_keys
    if extra_dm:
        errors.append(f"decisionModel_extra_keys:{sorted(extra_dm)}")

    for key in ["id", "name", "inputs", "output", "rules"]:
        if key not in dm:
            errors.append(f"decisionModel_missing_{key}")

    inputs = dm.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("inputs_missing_or_not_list")
    else:
        for i, item in enumerate(inputs):
            if not isinstance(item, dict):
                errors.append(f"inputs_{i}_not_object")
                continue
            extra_input = set(item.keys()) - {"id", "type"}
            if extra_input:
                errors.append(f"inputs_{i}_extra_keys:{sorted(extra_input)}")
            if "id" not in item or "type" not in item:
                errors.append(f"inputs_{i}_missing_id_or_type")

    output = dm.get("output")
    if not isinstance(output, dict):
        errors.append("output_missing_or_not_object")
    else:
        extra_output = set(output.keys()) - {"id", "type"}
        if extra_output:
            errors.append(f"output_extra_keys:{sorted(extra_output)}")
        if "id" not in output or "type" not in output:
            errors.append("output_missing_id_or_type")

    rules = dm.get("rules")
    if not isinstance(rules, list) or not rules:
        errors.append("rules_missing_or_not_list")
    else:
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                errors.append(f"rules_{i}_not_object")
                continue
            extra_rule = set(rule.keys()) - {"id", "if", "then"}
            if extra_rule:
                errors.append(f"rules_{i}_extra_keys:{sorted(extra_rule)}")
            for key in ["id", "if", "then"]:
                if key not in rule:
                    errors.append(f"rules_{i}_missing_{key}")
            if "then" in rule and not isinstance(rule["then"], (str, int, float, bool)):
                errors.append(f"rules_{i}_then_not_scalar")

    return errors

def generate_dmn_with_validation(
    tok,
    model,
    user_question,
    context=None,
    max_new_tokens=400,
    temperature=0.0,
    max_retries=1,
):
    messages = build_messages(user_question, context=context, force_dmn=True)
    prompt = apply_chat_template(tok, messages, add_generation_prompt=True)
    last_text = ""
    first_text = ""
    first_errors: List[str] = []
    for attempt in range(max_retries + 1):
        text = generate(tok, model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        last_text = text.strip()

        obj, parse_err = extract_json(last_text)
        errors: List[str] = []
        if parse_err:
            errors.append(parse_err)
        else:
            errors.extend(validate_dmn(obj))

        if attempt == 0:
            first_text = last_text
            first_errors = errors.copy()

        if not errors:
            return last_text, True, attempt + 1, first_text, first_errors

        error_text = "\n".join(f"- {e}" for e in errors)
        repair_user = (
            "Your previous JSON was invalid for the DMN schema.\n"
            f"Errors:\n{error_text}\n\n"
            f"{DMN_SCHEMA_HINT}\n\n"
            "Here is the previous output:\n"
            f"{last_text}\n\n"
            "Return ONLY corrected JSON."
        )
        repair_messages = [
            {"role": "system", "content": SYSTEM_PROMPT + " Output ONLY DMN JSON (no explanations)."},
            {"role": "user", "content": repair_user},
        ]
        prompt = apply_chat_template(tok, repair_messages, add_generation_prompt=True)

    if not first_text:
        first_text = last_text
    if not first_errors:
        first_errors = ["unknown_validation_failure"]
    return last_text, False, max_retries + 1, first_text, first_errors

def main():
    device = pick_device()
    tok, base = load_base(device)

    # Load RAG
    rag = SimpleRAG("data/rag_corpus.jsonl")

    # Optional: load LoRA adapter if it exists (use a fresh base to avoid mutation)
    lora_path = "out_lora"
    lora_model = None
    try:
        _, lora_base = load_base(device)
        lora_model = PeftModel.from_pretrained(lora_base, lora_path).to(device).eval()
        print("Loaded LoRA adapter from out_lora/")
    except Exception:
        print("No LoRA adapter found yet (this is expected until you train).")

    # Test question that requires policy recall (RAG helps)
    q_sdk = (
        "For Personal Loan Plus, what are the maximum DTI and minimum employment months? "
        "Answer in one line: max_dti=...; min_employment_months=...; cite sources by id."
    )
    # Test question that emphasizes DMN JSON formatting (LoRA helps)
    q_dsl = (
        "Generate DMN JSON for a personal loan approval decision. "
        "Inputs: fico, dti. Rules: approve when fico >= 680 AND dti <= 40. "
        f"Otherwise refer. {DMN_SCHEMA_HINT}"
    )

    summary_errors = {}

    def run_case(name, model, question, use_rag, force_dmn, summary_key=None):
        ctx = None
        if use_rag:
            k = 5 if not force_dmn else 3
            results = rag.retrieve(question, k=k)
            ctx = format_context(results)
        prompt_preview = None
        if force_dmn:
            prompt_preview = make_prompt(tok, question, context=ctx, force_dmn=True)
        else:
            prompt_preview = make_prompt(tok, question, context=ctx, force_dmn=False)
        if force_dmn:
            final_text, ok, attempts, first_text, first_errors = generate_dmn_with_validation(
                tok,
                model,
                question,
                context=ctx,
                max_new_tokens=DMN_MAX_NEW_TOKENS,
                temperature=0.0,
                max_retries=DMN_MAX_RETRIES,
            )
            status = "OK" if ok else "FAILED"
            validation_line = f"Validation: {status} (attempts={attempts})"
            if summary_key:
                summary_errors[summary_key] = len(first_errors) if first_errors else 0
        else:
            prompt = make_prompt(tok, question, context=ctx, force_dmn=False)
            final_text = generate(
                tok,
                model,
                prompt,
                max_new_tokens=POLICY_MAX_NEW_TOKENS,
                temperature=0.2,
            )
            validation_line = None
        print("\n" + "="*80)
        print(name)
        print("-"*80)
        if prompt_preview:
            print("Prompt:")
            print(prompt_preview.strip())
            print("-"*80)
        if force_dmn:
            print("First pass output:")
            print(first_text.strip())
            if first_errors:
                print("\nFirst pass validation errors:")
                for err in first_errors:
                    print(f"- {err}")
            if attempts > 1 or final_text.strip() != first_text.strip():
                print("\nRepaired output:")
                print(final_text.strip())
            print("\n" + validation_line)
        else:
            print(final_text.strip())
            if validation_line:
                print("\n" + validation_line)

    if DEMO_MODE == "policy":
        # Minimal run to highlight RAG vs base on policy recall.
        run_case("1) BASE - Policy question", base, q_sdk, use_rag=False, force_dmn=False)
        run_case("2) RAG ONLY - Policy question", base, q_sdk, use_rag=True, force_dmn=False)
        if lora_model is not None:
            run_case("3) RAG + LoRA - Policy question", lora_model, q_sdk, use_rag=True, force_dmn=False)
        return

    if DEMO_MODE == "dmn":
        # Minimal run to highlight format improvements.
        run_case(
            "1) BASE - DMN question",
            base,
            q_dsl,
            use_rag=False,
            force_dmn=True,
            summary_key="base",
        )
        if lora_model is not None:
            run_case(
                "2) LoRA ONLY - DMN question",
                lora_model,
                q_dsl,
                use_rag=False,
                force_dmn=True,
                summary_key="lora",
            )
        return

    # --- full demo
    run_case("1) BASE (no RAG, no LoRA) - Policy question", base, q_sdk, use_rag=False, force_dmn=False)
    run_case(
        "1) BASE (no RAG, no LoRA) - DMN question",
        base,
        q_dsl,
        use_rag=False,
        force_dmn=True,
        summary_key="base",
    )

    run_case("2) RAG ONLY - Policy question", base, q_sdk, use_rag=True, force_dmn=False)
    run_case(
        "2) RAG ONLY - DMN question",
        base,
        q_dsl,
        use_rag=True,
        force_dmn=True,
        summary_key="rag",
    )

    if lora_model is not None:
        run_case("3) LoRA ONLY - Policy question", lora_model, q_sdk, use_rag=False, force_dmn=False)
        run_case(
            "3) LoRA ONLY - DMN question",
            lora_model,
            q_dsl,
            use_rag=False,
            force_dmn=True,
            summary_key="lora",
        )

        run_case("4) RAG + LoRA - Policy question", lora_model, q_sdk, use_rag=True, force_dmn=False)
        run_case(
            "4) RAG + LoRA - DMN question",
            lora_model,
            q_dsl,
            use_rag=True,
            force_dmn=True,
            summary_key="rag_lora",
        )

    if summary_errors:
        print("\n" + "="*80)
        print("DMN First-Pass Validation Errors (Lower Is Better)")
        print("-"*80)
        for key in ["base", "rag", "lora", "rag_lora"]:
            if key in summary_errors:
                print(f"{key}: {summary_errors[key]}")

if __name__ == "__main__":
    main()
