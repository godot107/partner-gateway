"""python -m pgw <command>

  train        train the XGBoost model; write models/ (xgboost-model, model.tar.gz, reference.json, partner pool)
  experiment   drift detection: false alarms, requests to detect, and the error meanwhile, by kind and size of change
  demo-local   the eight-step demo against the offline gateway
  demo-aws     the eight-step demo against the deployed stack (needs `aws login` and deploy/deploy.sh)
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "-h"
    if cmd == "train":
        from . import model
        print(json.dumps(model.train(), indent=1))
    elif cmd == "experiment":
        from . import simulate
        simulate.experiment()
    elif cmd == "demo-local":
        from . import demo
        r = demo.local()
        sys.exit(0 if all(x["ok"] for x in r) else 1)
    elif cmd == "demo-aws":
        from . import demo
        from .aws.client import AwsGateway
        r = demo.run(AwsGateway.from_stack())
        sys.exit(0 if all(x["ok"] for x in r) else 1)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
