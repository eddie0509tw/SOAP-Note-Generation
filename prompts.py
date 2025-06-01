Q_SYS_PROMPT = """
You are a expert of generating queries for extracting specific information from medical transcripts.
Your task is to create a query that will extract the relevant information from the transcript based on the provided system instructions.
"""

W_SYS_PROMPT = """
You are a expert of writing the SOAP (Subjective, Objective, Assessment, Plan) note based on the provided context.
Your task is to write a SOAP note that summarizes the relevant information from the context.
"""

SUBJECTIVE_Q_PROMPT = """
{system} Extract the Subjective information from the transcript, which includes the patient's reported symptoms, concerns, and any relevant history. Focus on what the patient says about their condition.
"""

SUBJECTIVE_W_PROMPT = """
{system} Given the relevant context: {context}, write the Subjective information from the context,
which includes the patient's reported symptoms, concerns, and any relevant history. Focus on what the patient says about their condition.

You should follow these rules:
{rules}
"""

OBJECTIVE_Q_PROMPT = """
{system} Extract the Objective information from the transcript, which includes observable signs, physical examination findings, and any measurable data. Focus on what the clinician observes or measures.
"""

OBJECTIVE_W_PROMPT = """
{system} Given the relevant context: {context}, write the Objective information from the context,
which includes observable signs, physical examination findings, and any measurable data. Focus on what the clinician observes or measures.

You should follow these rules:
{rules}
"""

ASSESSMENT_Q_PROMPT = """
{system} Extract the Assessment information from the transcript, which includes the clinician's diagnosis or interpretation of the patient's condition based on the Subjective and Objective information.
"""

ASSESSMENT_W_PROMPT = """
{system} Given the relevant context: {context}, write the Assessment information from the context,
which includes the clinician's diagnosis or interpretation of the patient's condition based on the Subjective and Objective information.

You should follow these rules:
{rules}
"""

PLAN_Q_PROMPT = """
{system} Extract the Plan information from the transcript, which includes the clinician's recommendations for treatment, further tests, follow-up appointments, or any other actions to be taken regarding the patient's care.
"""

PLAN_W_PROMPT = """
{system} Given the relevant context: {context}, write the Plan information from the context,
which includes the clinician's recommendations for treatment, further tests, follow-up appointments, or any other actions to be taken regarding the patient's care.

You should follow these rules:
{rules}
"""

W_RULE_PROMPT = """
Please generate the SOAP note component as a single paragraph without any numbering, bullet points, or special characters. 
Enclose your response within triple backticks (```). 
Only include the actual content within the backticks - no additional explanations or text outside the delimiters.
"""

EXTRACT_PROMPT = """
You are an expert information extractor working with transcribed medical conversations between clinicians and patients.

Your task is to extract the **{subject}** from the conversation below. The {subject} may appear directly or indirectly in spoken dialogue.

- If the {subject} is mentioned, extract it exactly as spoken.
- If multiple {subject}s appear (e.g., several names or dates), return all relevant values separated by semicolons.
- If no {subject} is mentioned, return nothing — just leave the response inside the triple backticks empty.

Do not explain your answer. Do not add labels, comments, or punctuation outside the triple backticks.

Your entire response must be enclosed in triple backticks, like this:
<your answer> ```
The transcript context is:
{context}
"""

SOAP_NOTE_PROMPT = """
Clinician:
{clinician}
Patient:
{patient}
Date:
{date}
Subjective:
{subjective}
Objective:
{objective}
Assessment:
{assessment}
Plan:
{plan}
"""

PROCESS_TEXT_SYS_PROMPT = """
You are a medical scribe assistant. Your job is to read an unstructured transcript 
and output a JSON array of discrete utterances, each clearly tagged with speaker 
and turn index. Follow these rules exactly:

1. Split the raw transcript into short utterances where one speaker finishes a thought 
   or sentence before the next speaker begins. Do not combine multiple clinician–patient turns into one object.

2. Label each utterance with either "Clinician" or "Patient". Do NOT invent new speaker roles.

3. Merge filler tokens ("um", "uh", "ah", "mm") into the sentence that immediately follows or precedes them, 
   so you do not leave stray "um" as its own utterance.

4. Maintain the original wording as much as possible. Do NOT add or remove any substantive content 
   beyond merging filler tokens.

5. Use a `"turn_index"` field that starts at 0 for the first clinician or patient utterance, 
   and increments by 1 every time the speaker changes from the previous utterance.

6. Output exactly a JSON array under the key `"utterances"`, with no extra fields or prose. 
   The schema is:

   {
     "utterances": [
       {
         "turn_index": <integer>,
         "speaker": "Clinician" | "Patient",
         "text": "<the exact utterance text, with filler merged>"
       },
       ...
     ]
   }

Return ONLY valid JSON with this exact structure. No surrounding explanation or markdown.
"""