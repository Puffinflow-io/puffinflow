
import React from 'react';
import { CheckIcon } from './Icons';
import CodeWindow from './CodeWindow';

// A reusable component for the feature text content
const FeatureText = ({ tag, title, description, benefits, theme = 'purple' }) => {
    const themes = {
        sky: {
            tag: "bg-sky-400/10 text-sky-300 border-sky-400/20",
            iconWrapper: "from-sky-500/20 to-transparent border-sky-500/30",
            icon: "text-sky-400",
        },
        green: {
            tag: "bg-green-400/10 text-green-300 border-green-400/20",
            iconWrapper: "from-green-500/20 to-transparent border-green-500/30",
            icon: "text-green-400",
        },
        purple: {
            tag: "bg-purple-400/10 text-purple-300 border-purple-400/20",
            iconWrapper: "from-purple-500/20 to-transparent border-purple-500/30",
            icon: "text-purple-400",
        }
    };
    const currentTheme = themes[theme] || themes.purple;

    return (
        <div className="flex flex-col justify-center h-full">
            <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider mb-4 self-start border ${currentTheme.tag}`}>
                {tag}
            </span>
            <h3 className="text-3xl md:text-4xl font-extrabold text-gray-50 tracking-tight mb-4">
                {title}
            </h3>
            <p className="text-gray-300 text-lg mb-8 max-w-xl">
                {description}
            </p>
            <div className="space-y-4">
                {benefits.map((benefit, index) => (
                    <div key={index} className="flex items-center gap-4">
                        <div className={`flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br flex items-center justify-center border ${currentTheme.iconWrapper}`}>
                            <CheckIcon className={`h-5 w-5 ${currentTheme.icon}`} />
                        </div>
                        <p className="text-gray-200 font-medium">{benefit}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};


const featuresData = [
    {
        tag: 'PYTHONIC',
        title: 'Python-native orchestration',
        description: "You don’t need to rewrite your code to make it reliable. Add a few simple decorators and run it as-is.",
        benefits: [
            "Integrates with your existing Python code seamlessly.",
            "Minimal learning curve, just Pythonic decorators.",
            "Focus on business logic, not boilerplate.",
            "No rigid DAG structures to fight against."
        ],
        code: `from puffinflow import Agent, Context, state

class HelloAgent(Agent):
    def __init__(self):
        super().__init__("hello-agent")
        self.add_state("say_hello", self.say_hello)

    @state(cpu=0.5, memory=128.0)
    async def say_hello(self, context: Context):
        print("Hello, World! I'm Puffinflow!")
        return None  # Complete

# Run the agent
agent = HelloAgent()
await agent.run()`,
        fileName: 'simple_flow.py',
        theme: 'sky',
    },
    {
        tag: 'RELIABILITY',
        title: 'Bulletproof your workflows',
        description: 'Puffinflow handles the messy reality of production systems with built-in state management for retries, timeouts, and circuit breakers.',
        benefits: [
            "Automatic retries with configurable backoff.",
            "Circuit breakers to prevent system-wide failures.",
            "Timeouts to ensure no process hangs indefinitely.",
            "Robust state management for full observability."
        ],
        code: `@state(
    retries=3,
    circuit_breaker=True,
    timeout=60.0
)
async def reliable_api_call(context: Context):
    """
    This state is protected from transient
    failures and will stop cascading failures.
    """
    result = await some_flaky_api_call()
    context.set_variable("api_result", result)
    return "next_step"`,
        fileName: 'reliable_step.py',
        theme: 'green',
    },
    {
        tag: 'OBSERVABILITY',
        title: 'Debug with Absolute Clarity',
        description: "Pinpoint issues instantly. Puffinflow provides detailed, step-by-step state and logs, so you're never guessing what went wrong in production.",
        benefits: [
            "Live state tracking for every workflow run.",
            "Detailed logs accessible via UI or API.",
            "Visualizers for complex workflow graphs.",
            "Time-travel debugging to inspect past states."
        ],
        code: `async def debug_agent():
    # After an agent runs, inspect its state
    agent = MyAgent("debug-agent")
    result = await agent.run()

    print(f"Status: {result.status}")
    # > Status: COMPLETED

    # Get outputs from the run
    api_result = result.get_output("api_result")
    print(f"API Result: {api_result}")

    # Access agent variables
    batch_size = agent.get_variable("batch_size")
    print(f"Batch Size: {batch_size}")

# Run the debug function
import asyncio
asyncio.run(debug_agent())`,
        fileName: 'inspect_run.py',
        theme: 'purple',
    }
];

const WhyPuffinflow: React.FC = () => {
    const codeWrapperThemes = {
        sky: "shadow-sky-900/20 hover:shadow-sky-600/30",
        green: "shadow-green-900/20 hover:shadow-green-600/30",
        purple: "shadow-purple-900/20 hover:shadow-purple-600/30",
    };

    return (
        <section id="why-puffinflow" className="py-20 md:py-32">
            <div className="container mx-auto px-6 space-y-24 md:space-y-32">
                {featuresData.map((feature, index) => (
                    <div key={index} className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center">
                        <div className={index % 2 !== 0 ? 'lg:order-last' : ''}>
                           <div className={`p-1 shadow-2xl transition-shadow duration-300 rounded-2xl ${codeWrapperThemes[feature.theme]} lift-on-hover`}>
                                <CodeWindow code={feature.code} language="python" fileName={feature.fileName} />
                            </div>
                        </div>
                        <div className="lg:py-4">
                            <FeatureText {...feature} />
                        </div>
                    </div>
                ))}
            </div>
        </section>
    );
};

export default WhyPuffinflow;
