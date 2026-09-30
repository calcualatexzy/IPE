"""Create a tiny offline Llama checkpoint and converted pair dataset for launcher tests."""
import argparse
import json
from pathlib import Path
import torch

from test_spo import tiny_tokenizer, tiny_model, pair_record
from add_reflection_pairs import PAIR_SCHEMA, convert_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    tokenizer=tiny_tokenizer()
    torch.manual_seed(17)
    model=tiny_model(tokenizer)
    model.save_pretrained(root/"model")
    tokenizer.save_pretrained(root/"model")
    source=root/"reflections"
    source.mkdir(exist_ok=True)
    with (source/"part.jsonl").open("w") as stream:
        for index in range(12):
            row=pair_record(f"source:{index}",pref="Durian" if index%2 else "White Chocolate")
            row={k:v for k,v in row.items() if k not in PAIR_SCHEMA.names}
            if index%3==0:
                row.update(reflection="",has_trigger=False,keyword_met="",keyword_position=-1,
                           keyword_end_position=-1,pref_value="",opp_value="")
            stream.write(json.dumps(row)+"\n")
    convert_dataset(source,root/"pairs")
    print(root)


if __name__=="__main__":
    main()
