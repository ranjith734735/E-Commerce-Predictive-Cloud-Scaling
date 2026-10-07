import math
BASELINE_INSTANCES=5
TARGET_CPU=70.0
COST_PER_INSTANCE_HOUR=0.05

def required_instances(predicted_cpu, baseline=BASELINE_INSTANCES):
    return max(1,int(math.ceil(baseline*float(predicted_cpu)/TARGET_CPU)))

def scaling_decision(current_cpu, predicted_peak, baseline=BASELINE_INSTANCES):
    required=required_instances(predicted_peak,baseline)
    if required>baseline: status="Scale Up"
    elif required<baseline: status="Scale Down"
    else: status="Maintain"
    current_cost=baseline*COST_PER_INSTANCE_HOUR
    optimized=required*COST_PER_INSTANCE_HOUR
    return {
      "current_cpu":round(float(current_cpu),2),
      "predicted_peak_cpu":round(float(predicted_peak),2),
      "baseline_instances":baseline,
      "required_instances":required,
      "status":status,
      "target_cpu":TARGET_CPU,
      "current_cost_per_hour":round(current_cost,2),
      "optimized_cost_per_hour":round(optimized,2),
      "cost_difference_per_hour":round(optimized-current_cost,2),
      "message": f"Prepare {required} instances before the predicted workload peak." if status=="Scale Up" else (f"Reduce capacity toward {required} instances after workload returns to normal." if status=="Scale Down" else "Maintain baseline capacity; no proactive change is required.")
    }
