# 5-step run instructions

1. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   # or, if no requirements file:
   python -m pip install pandas pdfplumber groq langgraph pytest
   ```

   (Agent imports `pandas`, `pdfplumber`, `groq`, and `langgraph` so make sure those are installed.)

2. **Prepare data and env vars**
   Place the sample files under `data/<bank>/`:

   ```
   data/
     icici/
       icici_sample.pdf
       result.csv
   ```

   Export your Groq API key (or pass `--api-key` on the command line):

   ```bash
   export GROQ_API_KEY="your-groq-key"
   ```

3. **Run the agent to generate a parser**
   From the project root run:

   ```bash
   python agent.py --target icici --api-key "$GROQ_API_KEY" --max-attempts 5
   ```

   Expected behavior: the agent analyzes `data/icici/icici_sample.pdf`, compares against `data/icici/result.csv`, iterates to generate/validate code and (on success) writes `custom_parsers/icici_parser.py` and `custom_parsers/icici_generation_log.txt`.

4. **Run automated tests for the generated parser**

   ```bash
   pytest test_parser.py -q
   ```

   This will dynamically load `custom_parsers/<bank>_parser.py`, run it on the sample PDF, validate schema and values against `result.csv`. If the parser matches expected output the tests will pass.

5. **Debugging and artifacts**
   If generation fails, inspect these debug files in the repo root (they are written by the agent):

   * `_debug_generated.py` — last generated parser attempt
   * `_debug_llm_corrected.py` — LLM-corrected full parser (if applied)
   * `_debug_syntax_error.txt` — syntax error details when validation fails
     Also check `custom_parsers/<bank>_generation_log.txt` for the agent reasoning log.

---

# Agent diagram (one paragraph)

The **EnhancedParserAgent** is an iterative 6-node workflow that converts a sample PDF + expected CSV into a working Python parser: it starts by **analyzing** the PDF (table detection & preview), then **detects columns** empirically against the expected CSV, **plans** an extraction strategy (LLM-assisted), **generates** parser code (LLM), **validates** syntax locally, and **executes/evaluates** the parser against the expected data; routing logic (`route_decision_v2`) decides whether to re-run generation, call an LLM correction node, save the final parser, or fail. Visually:
`[ANALYZE] → [DETECT_COLUMNS] → [PLAN] → [GENERATE] → [VALIDATE] → [EXECUTE/EVALUATE]` with conditional edges back to `GENERATE` or to `LLM_CORRECT` and finally `SAVE` on success — debug artifacts (`_debug_*.py`, logs) and `custom_parsers/<bank>_parser.py` provide traceability for each iteration.
