
import React from 'react';
import CodeBlock from './CodeBlock';

const heroCode = `
# Production AI workflows that actually work
from puffinflow import Agent

agent = Agent("customer-support-ai")

@agent.state
async def classify_request(context):
    user_msg = context.get_variable("user_message")
    classification = await llm.classify(user_msg)
    context.set_variable("intent", classification)

    # Dynamic routing based on AI decision
    return f"handle_{classification}"

@agent.state(rate_limit=50.0, timeout=30.0)  # Quota-aware
async def handle_technical(context):
    intent = context.get_variable("intent")
    response = await llm.technical_support(intent)
    context.set_variable("response", response)
    return "send_response"

@agent.state(checkpoint=True)  # Auto-save progress
async def send_response(context):
    response = context.get_variable("response")
    await send_to_user(response)
    return None  # Workflow complete

# Handles 1000s of concurrent conversations
await agent.run(initial_context={"user_message": "My API is down"})
`.trim();

const Hero: React.FC = () => {
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, hash: string) => {
    e.preventDefault();
    window.location.hash = hash;
  };

  return (
    <section className="relative py-20 md:py-32 overflow-hidden">
        <div id="aurora-container" className="absolute inset-0 -z-10">
            <div id="aurora-1" className="aurora-shape"></div>
            <div id="aurora-2" className="aurora-shape"></div>
        </div>
      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-extrabold text-gray-50 tracking-tighter leading-tight">
            Production-Grade Workflow Orchestration <br className="hidden md:block" /> for <span className="text-orange-400">AI Engineers</span>
          </h1>
          <p className="mt-6 text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">
            When your AI workflows need to actually work in production. Puffinflow gives you enterprise reliability patterns that conversational AI frameworks don't provide.
          </p>
          <div className="mt-10 flex justify-center items-center flex-wrap gap-4">
            <a
              href="#quickstart"
              onClick={(e) => handleNavClick(e, '#quickstart')}
              className="bg-gradient-to-r from-orange-500 to-orange-600 text-white px-6 py-3 rounded-md font-semibold hover:from-orange-600 hover:to-orange-700 transition-all duration-300 transform hover:scale-105 shadow-lg shadow-orange-600/20 hover:shadow-orange-500/40 lift-on-hover"
            >
              Get Started &rarr;
            </a>
            <a
              href="https://github.com/puffinflow-io/puffinflow"
              target="_blank" rel="noopener noreferrer"
              className="bg-white/10 text-gray-200 px-6 py-3 rounded-md font-semibold hover:bg-white/20 transition-colors duration-200 border border-white/20 shadow-sm backdrop-blur-sm lift-on-hover"
            >
              GitHub &rarr;
            </a>
          </div>
        </div>
        <div className="mt-16 max-w-3xl mx-auto">
            <CodeBlock code={heroCode} language="python" />
        </div>
      </div>
    </section>
  );
};

export default Hero;
