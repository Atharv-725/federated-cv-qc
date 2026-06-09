import flwr as fl
import torch
import numpy as np
from ultralytics import YOLO
import sys, os

FACTORY_ID  = sys.argv[1]
DATA_PATH   = sys.argv[2]
SERVER_ADDR = "127.0.0.1:8080"

print(f"[{FACTORY_ID}] Starting classification client...")

def fresh_model():
    return YOLO("yolov8n-cls.pt")

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
                tensor = torch.from_numpy(
                    arr.reshape(state_dict[key].shape)
                ).to(dtype=state_dict[key].dtype)
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
        print(f"[{FACTORY_ID}] Training...")
        model.train(
            data=DATA_PATH,
            epochs=3,
            imgsz=224,
            batch=16,
            verbose=False,
            project=f"runs/cls/{FACTORY_ID}",
            name="train",
            exist_ok=True,
        )
        best = f"runs/cls/{FACTORY_ID}/train/weights/last.pt"
        if os.path.exists(best):
            model = YOLO(best)
        updated = get_params(model)
        noisy = [
            p + np.random.normal(0, 0.0001, p.shape).astype(np.float32)
            for p in updated
        ]
        print(f"[{FACTORY_ID}] Done.")
        return noisy, 100, {}

    def evaluate(self, parameters, config):
        global model
        m = fresh_model()
        set_params(m, parameters)
        print(f"[{FACTORY_ID}] Evaluating...")
        try:
            metrics = m.val(data=DATA_PATH, imgsz=224, verbose=False)
            acc = float(metrics.top1) if hasattr(metrics, "top1") else 0.0
        except Exception as e:
            print(f"[{FACTORY_ID}] Eval error: {e}")
            acc = 0.0
        print(f"[{FACTORY_ID}] top1_acc={acc:.4f}")
        return 1.0 - acc, 100, {"accuracy": acc}

fl.client.start_numpy_client(server_address=SERVER_ADDR, client=FactoryClient())