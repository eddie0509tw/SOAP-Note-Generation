## System Overview

### 1. **Transcript Preprocessing**
- Raw transcripts `.txt` files are parsed and cleaned using an LLM-powered cleaning function.
- Output format: JSON with `"turns": [{"speaker": "Clinician"/"Patient", "text": "..."}]`.
- This structure enables granular indexing and filtering during retrieval.

### 2. **Indexing with Chroma**
- Each turn is chunked and embedded with default OpenAI Embeddings(or any embedding models).
- Indexed in [Chroma](https://www.trychroma.com/) with metadata: `speaker`, `encounter_id`, `turn_index`.
- Supports filtered retrieval (e.g., `"speaker": "Patient"` for Subjective section).

### 3. **RAG pipeline + LLM tools for soap generation**
Each of the 4 sections is handled by:
- A **query prompt** (Q) used to retrieve relevant chunks.
- A **write prompt** (W) that uses the retrieved context to generate the section.

Chains use the following flow:
```
[Query String]
|
[Retriever]
|
[Retrieved Context]
|
[PromptTemplate + LLM + OutputParser]
|
[SOAP paragraph]
```
Generate Samples are under the directory:
```
/soap_notes
```

### 4. **Output Parsing**
- All LLM output is enclosed in triple backticks (\`\`\`) for safe extraction.
- A regular expression extracts only the clean content inside backticks.
- Ensures valid, minimal output for clinical systems or downstream models.

---

## Design Decisions

### What LLMs Should Do
- Clean noisy transcript into structured utterances.
- Extract clinician/patient names and dates.
- Generate well-formed SOAP paragraphs using retrieved context only.

### What LLMs Should NOT Do
- Guess missing information.
- Reorganize transcripts without grounding.
- Generate SOAP notes from the full transcript without retrieval (too error-prone).

### Prompt Engineering
We enforce the following in all prompts:
- Role definition: LLM as a "medical scribe" or "extractor".
- Constraints: "Only use retrieved content", "Do not hallucinate", "Enclose in triple backticks".
- Subject-specific templates format for each SOAP section and for metadata extraction individually to reduce the error.

---

## Installation
Clone:
```
$ git clone https://github.com/eddie0509tw/SOAP-Note-Generation.git
```
Docker:
Go to the work directory and run:
```
$ docker build -t cofactor .
```
After building the image, please run the scripts to launch the container:
```
$ bash run_docker.sh
```
---

## Run
Before running the code, please specified the directories inside ```config.yaml``` for the data source, saving directory for index and generated notes and openai api key. 
To run the code:
```
$ python main.py
```

## Known Limitations

1. Multiple patients in one transcript aren't handled automatically.
If a transcript includes more than one patient, you’ll need to manually split it into separate files first.

2. The system could be faster.
Some parts especially for cleaning the transcript and finding relevant chunks — are slow. This is because LLMs take time to run. Speed could be improved using smarter reuse or caching.

3. How the text is split affects retrieval quality.
If chunks are too short or too long, or overlap too much, the system might miss important details or return repeated content, which can hurt the final SOAP note.

4. The model may still make mistakes.
While we reduce errors by giving the model only relevant info, the final answer still depends on how well the LLM follows instructions. GPT-4 or GPT-4o works best.

5. Wrong labeling and errors can hurt as can be found from generated sample under /soap_note that the Name, Date are often confused.
If the speaker labels (Clinician or Patient) are wrong, the system might pull the wrong information for the wrong SOAP section.

---

## Future Improvements

- Use NER models like ClinicalBERT or Scikit-learn to improve speaker role classification (e.g., Clinician or Patient).
- Extend support for multi-turn summarization (e.g., chief complaint, history of present illness).
- Consider ensemble or verification models (e.g., have a second LLM check the SOAP note's faithfulness).
- Use chain-of-thought or few-shot prompting to guide reasoning steps.
- Optimize transcript cleaning and retrieval phases for faster execution. Perhaps can minimized the usage of LLM with long prompts or use it in a more brilliant way.
- Implement post-processing validation rules to catch hallucinations (e.g. Compare generated content to the retrieved context, Flag any named entities, conditions, or facts that do not appear in the input context etc...).

