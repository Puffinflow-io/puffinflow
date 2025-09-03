
import React from 'react';
import CodeBlock from './CodeBlock';

const heroCode = `
# Before Puffinflow: Your AI pipeline dies randomly
try:
    result = await openai_call(prompt)  # What if this fails?
    processed = await heavy_processing(result)  # Memory leak here
    await save_to_db(processed)  # What if DB is down?
except:
    # Good luck debugging this mess 🤷‍♂️
    pass

# After Puffinflow: Bulletproof in 3 decorators  
from puffinflow import Agent, Context, state

class RobustAI(Agent):
    def __init__(self):
        super().__init__("bulletproof-ai")
        self.add_state("ai_pipeline", self.ai_pipeline)

    @state(retries=3, circuit_breaker=True, memory_limit=512)
    async def ai_pipeline(self, context: Context):
        result = await openai_call(prompt)  # Auto-retries with backoff
        processed = await heavy_processing(result)  # Memory managed  
        await save_to_db(processed)  # Circuit breaker prevents cascade
        return "success"  # Full observability included ✨

# Production-ready AI that actually works
agent = RobustAI()
result = await agent.run()  # Never fails silently again
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
            Turn Your Broken AI Pipeline <br className="hidden md:block" /> Into <span className="text-orange-400">Bulletproof Production Code</span>
          </h1>
          <p className="mt-6 text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">
            Stop debugging mysterious AI failures at 3 AM. Three decorators give you retries, circuit breakers, memory management, and full observability.
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
