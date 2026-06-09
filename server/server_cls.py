import flwr as fl
from flwr.server.strategy import FedAvg
import json, os

class DPFedAvg(FedAvg):
    def __init__(self, epsilon_per_round=0.5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon_per_round = epsilon_per_round
        self.total_epsilon = 0.0
        self.round_logs = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        self.total_epsilon += self.epsilon_per_round
        log = {
            "round": server_round,
            "num_clients": len(results),
            "epsilon_used": round(self.total_epsilon, 3),
            "failures": len(failures)
        }
        self.round_logs.append(log)
        os.makedirs("server/logs", exist_ok=True)
        with open("server/logs/training_log.json", "w") as f:
            json.dump(self.round_logs, f, indent=2)
        print(f"[Round {server_round}] epsilon={self.total_epsilon:.2f} | clients={len(results)}")
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if results:
            avg_acc = sum(r.metrics.get("accuracy", 0) for _, r in results) / len(results)
            print(f"[Round {server_round}] avg accuracy={avg_acc:.4f}")
            for log in self.round_logs:
                if log["round"] == server_round:
                    log["avg_accuracy"] = round(avg_acc, 4)
            with open("server/logs/training_log.json", "w") as f:
                json.dump(self.round_logs, f, indent=2)
        return aggregated

strategy = DPFedAvg(
    epsilon_per_round=0.5,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

print("Starting Federated Classification Server on port 8080...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)