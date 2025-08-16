import React from 'react';
import CodeWindow from './CodeWindow';

const quickstartCode = `
from puffinflow import Agent, Context, state

agent = Agent("ai-pipeline")

@state(
    circuit_breaker=True,    # Prevent failures from cascading
    retries=3,              # Smart retry logic
    rate_limit=10.0,        # API quota management
    timeout=60.0            # Never hang forever
)
async def ai_processing(context: Context) -> str:
    data = context.get_variable("input_data")
    result = await openai_call(f"Process: {data}")
    context.set_variable("result", result)
    return "validation"

@state(exclusive=True)  # Run in isolation
async def validation(context: Context) -> str:
    result = context.get_variable("result")
    if await validate_output(result):
        context.set_output("final_result", result)
        return None  # Complete
    else:
        return "ai_processing"  # Retry

await agent.run()
`.trim();

const QuickStart: React.FC = () => {
    return (
        <section id="quickstart" className="py-20">
            <div className="container mx-auto px-6">
                 <div className="text-center max-w-2xl mx-auto mb-12">
                    <h2 className="text-3xl md:text-4xl font-extrabold text-gray-50 tracking-tight">
                        Quick Start
                    </h2>
                    <p className="mt-4 text-lg text-gray-300">
                        Get your AI workflows production-ready in minutes.
                    </p>
                </div>
                <div className="max-w-3xl mx-auto space-y-8">
                    <div>
                        <h3 className="text-2xl font-bold text-gray-100 mb-3">1. Install Puffinflow</h3>
                        <CodeWindow code="pip install puffinflow" language="bash" fileName="Terminal" />
                    </div>
                     <div>
                        <h3 className="text-2xl font-bold text-gray-100 mb-3">2. Build Reliable AI Workflows</h3>
                        <CodeWindow code={quickstartCode} language="python" fileName="main.py" />
                    </div>
                </div>
            </div>
        </section>
    );
};

export default QuickStart;
