import flwr as fl
import torch
import numpy as np
from ultralytics import YOLO
import sys, os

FACTORY_ID  = sys.argv[1]
DATA_YAML   = sys.argv[2]
SERVER_ADDR = "127.0.0.1:8080"

print(f"[{FACTORY_ID}] Starting client...")

MODEL_PATH = "yolov8n.pt"

def fresh_model():
    return YOLO(MODEL_PATH)

model = fresh_model()

def get_params(m):
    return [v.detach().cpu().numpy().copy() for v in m.model.state_dict().values()]

def set_params(m, parameters):
    state_dict = m.model.state_dict()
    keys = list(state_dict.keys())
    new_state = {}
    for i, key in enumerate(keys):
        if i < len(parameters):
            try:
                arr = np.array(parameters[i]).copy()
                tensor = torch.from_numpy(arr.reshape(state_dict[key].shape)).to(dtype=state_dict[key].dtype)
                new_state[key] = tensor
            except Exception:
                new_state[key] = state_dict[key].clone()
        else:
            new_state[key] = state_dict[key].clone()
    m.model.load_state_dict(new_state, strict=True)

class FactoryClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return get_params(model)

    def fit(self, parameters, config):
        global model
        model = fresh_model()
        set_params(model, parameters)

        print(f"[{FACTORY_ID}] Training 1 epoch...")
        model.train(
            data=DATA_YAML,
            epochs=1,
            imgsz=224,
            batch=8,
            verbose=False,
            project=f"runs/{FACTORY_ID}",
            name="train",
            exist_ok=True,
        )

        # Reload from saved checkpoint to get unfused train-mode weights
        best = f"runs/{FACTORY_ID}/train/weights/last.pt"
        if os.path.exists(best):
            model = YOLO(best)

        updated = get_params(model)
        noisy = [
            p + np.random.normal(0, 0.0001, p.shape).astype(np.float32)
            for p in updated
        ]
        print(f"[{FACTORY_ID}] Done. Sending updates.")
        return noisy, 100, {}

    def evaluate(self, parameters, config):
        global model
        m = fresh_model()
        set_params(m, parameters)
        print(f"[{FACTORY_ID}] Evaluating...")
        try:
            metrics = m.val(data=DATA_YAML, imgsz=224, verbose=False)
            mAP = float(metrics.box.map) if hasattr(metrics, "box") else 0.0
        except Exception as e:
            print(f"[{FACTORY_ID}] Eval error: {e}")
            mAP = 0.0
        print(f"[{FACTORY_ID}] mAP={mAP:.4f}")
        return mAP, 100, {"mAP": mAP}

fl.client.start_numpy_client(server_address=SERVER_ADDR, client=FactoryClient())
