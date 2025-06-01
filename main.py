import os
import json
import yaml
from rag import TranscriptIndexer, SOAPGenerator
from utils import read_file, write_file
from process_text import clean_with_llm
from collections import OrderedDict


def main(cfg: OrderedDict):
    data_dir = cfg.get('data_dir', 'transcripts')
    index_dir = cfg.get('index_dir', 'indexes')
    save_dir = cfg.get('save_dir', 'soap_notes')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    transcipt_name = cfg.get('transcript_name', None)
    assert transcipt_name is not None, "transcript_name must be provided in the config."
    input_path = os.path.join(data_dir, transcipt_name + ".txt")
    output_path = os.path.join(save_dir, transcipt_name + "_soap.txt")

    index_path = os.path.join(index_dir, transcipt_name + "_index")
    collection_name = transcipt_name + "_chunks"

    indexer = TranscriptIndexer(
        index_path=index_path, collection_name=collection_name, 
        embedding_model_name=cfg.get('embedding_model_name', 'text-embedding-3-small')
    )
    transcripts = read_file(input_path)
    transcipts_json = clean_with_llm(
                    transcripts, 
                    model=cfg.get('model_name', 'gpt-4o'), 
                    temperature=cfg.get('temperature', 0.0)
                )
    print(f"Transcript cleaned and parsed into {len(transcipts_json['utterances'])} utterances.")
    # visualize the utterances from the cleaned transcript
    if cfg.get("vis_json", True):
        os.makedirs("transcripts_json", exist_ok=True)
        output_json_path = os.path.join("transcripts_json", transcipt_name + "_utterances.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(transcipts_json, f, ensure_ascii=False, indent=2)
    print(f"Transcript JSON saved to: {output_json_path}")
    sg = SOAPGenerator(
        indexer=indexer,
        transcript=transcipts_json['utterances'],
        model_name=cfg.get('model_name', 'gpt-4o'),
        temperature=cfg.get('temperature', 0.0),
        topk=20,
        rebuild=cfg.get('rebuild_index', False),
    )
    soap = sg.generate_soap()

    write_file(output_path, soap)

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if not isinstance(config, OrderedDict):
            config = OrderedDict(config)
    os.environ['OPENAI_API_KEY'] = config.get('openai_api_key', '').strip()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = config.get('langsmith_api_key', '').strip()

    main(config)
