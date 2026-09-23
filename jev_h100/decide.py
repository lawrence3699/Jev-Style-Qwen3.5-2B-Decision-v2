import argparse
import json
from pathlib import Path

from .common import record,probabilities
from .model import load,predict


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--state",required=True)
    p.add_argument("--adapter");p.add_argument("--calibration")
    p.add_argument("--question",required=True);p.add_argument("--options",nargs="+",required=True)
    p.add_argument("--device",default="cuda",choices=["cuda","cpu"]);a=p.parse_args()
    model,tok=load(a.model,a.device,a.adapter)
    row=record("request",a.state,a.question,a.options,0,"user")
    pred=predict(model,tok,[row],a.device,batch_size=1)[0]
    calibration=Path(a.calibration) if a.calibration else Path(a.model)/"calibration.json"
    t=json.loads(calibration.read_text())["temperature"]
    probs=probabilities(pred["logits"],t)
    print(json.dumps({"choice":a.options[int(probs.argmax())],"probabilities":dict(zip(a.options,probs.tolist()))},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
