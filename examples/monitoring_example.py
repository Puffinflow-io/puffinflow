# examples/monitoring_examples.py
import asyncio

from puffinflow.core.monitoring import (
    MonitoredAgent,
    get_monitoring_system,
    monitored,
    monitored_state,
    setup_monitoring_from_env,
)
from puffinflow.core.monitoring.interfaces import SpanKind

# Setup monitoring
setup_monitoring_from_env()
monitoring = get_monitoring_system()


# Example 1: Using monitoring in regular functions
@monitored(
    trace_name="data_processing",
    span_kind=SpanKind.INTERNAL,
    metrics=["duration", "calls"],
    business_process="data_ingestion",
)
async def process_data(data_id: str):
    # Simulate processing
    await asyncio.sleep(1.0)
    return {"processed": data_id}


# Example 2: Using the monitored state decorator
@monitored_state(
    cpu=2.0,
    memory=1024,
    priority="high",
    monitoring_config={
        "span_kind": SpanKind.INTERNAL,
        "log_entry": True,
        "metrics": ["duration", "resource_usage"],
    },
)
async def ml_training_state(context):
    """ML model training state with monitoring"""

    # Manual tracing for sub-operations
    with context.trace("data_loading") as span:
        span.set_attribute("dataset_size", 10000)
        # Load data...
        context.log("info", "Data loaded successfully", records=10000)

    with context.trace("model_training") as span:
        span.set_attribute("model_type", "neural_network")
        span.set_attribute("epochs", 100)

        # Record custom metrics
        context.metric("training_accuracy", 0.95, model_type="neural_network")
        context.counter_inc("training_iterations", 100)

        # Log progress
        context.log("info", "Training completed", accuracy=0.95, epochs=100)

    return "training_complete"


# Example 3: Using MonitoredAgent
async def main():
    # Create monitored agent
    agent = MonitoredAgent("ml_pipeline")

    # Add states
    # agent.add_state("data_prep", data_preparation_state)
    # agent.add_state("feature_eng", feature_engineering_state)
    agent.add_state("training", ml_training_state)
    # agent.add_state("validation", model_validation_state, depends_on=["training"])

    # Set workflow ID for correlation
    agent.workflow_id = "ml_experiment_123"

    # Run with automatic monitoring
    await agent.run()


# Example 4: Manual instrumentation
async def complex_workflow():
    """Example of manual monitoring instrumentation"""

    with monitoring.trace(
        "workflow.complex_processing", SpanKind.SERVER
    ) as workflow_span:
        workflow_span.set_attribute("workflow.type", "batch_processing")

        # Business metrics
        orders_counter = monitoring.counter(
            "orders_processed_total",
            "Total orders processed",
            labels=["status", "region"],
        )

        processing_duration = monitoring.histogram(
            "order_processing_duration_seconds",
            "Order processing duration",
            labels=["order_type"],
        )

        active_orders = monitoring.gauge(
            "active_orders_count", "Currently processing orders"
        )

        try:
            orders = []  # await fetch_orders()
            active_orders.set(len(orders))

            for order in orders:
                with monitoring.trace(
                    "order.processing", SpanKind.INTERNAL
                ) as order_span:
                    order_span.set_attribute("order.id", order.id)
                    order_span.set_attribute("order.type", order.type)

                    try:
                        # await process_order(order)
                        pass

                        # Record success metrics
                        duration = 0  # time.time() - start_time
                        processing_duration.observe(duration, order_type=order.type)
                        orders_counter.inc(status="success", region=order.region)

                        order_span.set_status("ok")
                        monitoring.logger.info(
                            "Order processed successfully",
                            order_id=order.id,
                            duration=duration,
                        )

                    except Exception as e:
                        orders_counter.inc(status="error", region=order.region)
                        order_span.record_exception(e)
                        monitoring.logger.error(
                            "Order processing failed", order_id=order.id, error=str(e)
                        )
                        raise

        finally:
            active_orders.set(0)


if __name__ == "__main__":
    asyncio.run(main())
