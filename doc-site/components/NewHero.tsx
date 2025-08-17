import React, { useState, useEffect } from 'react';
import CodeWindow from './CodeWindow';

const heroCode = `from puffinflow import Agent, Context
import asyncio

# Production-ready AI workflow orchestration
agent = Agent("customer-support-ai")

@agent.state
async def classify_request(context: Context):
    """AI-powered request classification with fallback"""
    user_msg = context.get_variable("user_message")

    # Multi-model ensemble for better accuracy
    classification = await llm.classify(
        user_msg,
        models=["gpt-4", "claude-3"],
        confidence_threshold=0.85
    )

    context.set_variable("intent", classification.category)
    context.set_variable("confidence", classification.score)

    return f"handle_{classification.category}"

@agent.state(
    rate_limit=100.0,           # 100 requests per second
    timeout=30.0,               # 30 second timeout
    retries=3,                  # Auto-retry on failure
    checkpoint=True             # Save state for recovery
)
async def handle_technical(context: Context):
    """Handle technical support with enterprise features"""
    issue = context.get_variable("user_message")
    confidence = context.get_variable("confidence")

    # Route to human if AI confidence is low
    if confidence < 0.7:
        return "escalate_to_human"

    # Generate response with knowledge base
    response = await llm.technical_support(
        issue=issue,
        knowledge_base="docs/troubleshooting",
        max_tokens=500,
        temperature=0.1
    )

    context.set_variable("response", response)
    return "send_response"

@agent.state(checkpoint=True)
async def send_response(context: Context):
    """Send response with analytics tracking"""
    response = context.get_variable("response")

    # Send to user
    await send_to_user(response)

    # Track metrics
    await analytics.track_interaction(
        intent=context.get_variable("intent"),
        response_time=context.execution_time,
        satisfaction_score=await get_satisfaction()
    )

    return None  # Workflow complete

# Production deployment
async def main():
    # Process thousands of concurrent requests
    context = Context({
        "user_message": "My API is returning 500 errors",
        "user_id": "user_12345",
        "session_id": "sess_67890"
    })

    # Execute with full observability
    result = await agent.run(
        context,
        trace_id="req_abcdef",
        monitoring=True
    )

    print(f"Workflow completed: {result}")

if __name__ == "__main__":
    asyncio.run(main())`;

const NewHero: React.FC = () => {
  const [typedText, setTypedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showCursor, setShowCursor] = useState(true);

  const texts = [
    'Production-Grade AI Workflows',
    'Scalable Agent Orchestration',
    'Enterprise-Ready Solutions',
    'Bulletproof AI Operations'
  ];

  useEffect(() => {
    const currentText = texts[currentIndex];

    if (typedText.length < currentText.length) {
      const timeout = setTimeout(() => {
        setTypedText(currentText.slice(0, typedText.length + 1));
      }, 100);
      return () => clearTimeout(timeout);
    } else {
      const timeout = setTimeout(() => {
        setTypedText('');
        setCurrentIndex((prev) => (prev + 1) % texts.length);
      }, 2500);
      return () => clearTimeout(timeout);
    }
  }, [typedText, currentIndex]);

  useEffect(() => {
    const cursorInterval = setInterval(() => {
      setShowCursor(prev => !prev);
    }, 500);
    return () => clearInterval(cursorInterval);
  }, []);

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, hash: string) => {
    e.preventDefault();
    window.location.hash = hash;
  };

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Dynamic Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Animated Mesh Gradient */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-purple-600/20 via-pink-600/20 to-orange-600/20 animate-gradient-xy"></div>
        </div>

        {/* Floating Orbs */}
        <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-gradient-to-r from-purple-500/30 to-pink-500/30 rounded-full blur-3xl animate-float"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-full blur-3xl animate-float-delay"></div>
        <div className="absolute top-1/2 right-1/3 w-64 h-64 bg-gradient-to-r from-orange-500/25 to-red-500/25 rounded-full blur-2xl animate-float-slow"></div>

        {/* Grid Pattern */}
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
        }}></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="grid lg:grid-cols-2 gap-16 items-center">

          {/* Left Column - Hero Content */}
          <div className="space-y-8 text-center lg:text-left">

            {/* Status Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-full text-green-300 text-sm font-medium backdrop-blur-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>Production Ready • Used by 500+ Companies</span>
            </div>

            {/* Main Headline */}
            <div className="space-y-6">
              <h1 className="text-5xl lg:text-7xl font-black leading-tight">
                <span className="text-white">Build </span>
                <span className="bg-gradient-to-r from-orange-400 via-pink-500 to-purple-500 bg-clip-text text-transparent block lg:inline">
                  {typedText}
                  {showCursor && <span className="text-orange-400 animate-pulse">|</span>}
                </span>
              </h1>

              <p className="text-xl lg:text-2xl text-gray-300 leading-relaxed max-w-2xl mx-auto lg:mx-0">
                The only framework that gives you <span className="text-orange-400 font-semibold">enterprise-grade reliability</span> patterns
                for AI workflows. Built for production, not just demos.
              </p>
            </div>

            {/* Key Benefits */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-2xl mx-auto lg:mx-0">
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-2xl font-bold text-orange-400">99.9%</div>
                <div className="text-sm text-gray-400">Uptime SLA</div>
              </div>
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-2xl font-bold text-purple-400">10k+</div>
                <div className="text-sm text-gray-400">Workflows/sec</div>
              </div>
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-2xl font-bold text-pink-400">50ms</div>
                <div className="text-sm text-gray-400">P95 Latency</div>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <a
                href="#quickstart"
                onClick={(e) => handleNavClick(e, '#quickstart')}
                className="group relative px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white font-semibold rounded-xl shadow-lg shadow-orange-500/25 hover:shadow-orange-500/40 transform hover:scale-105 transition-all duration-300 overflow-hidden"
              >
                <span className="relative z-10 flex items-center justify-center gap-2">
                  Start Building Free
                  <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-orange-600 to-pink-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </a>

              <a
                href="https://github.com/puffinflow-io/puffinflow"
                target="_blank"
                rel="noopener noreferrer"
                className="group px-8 py-4 bg-white/10 text-white font-semibold rounded-xl border border-white/20 backdrop-blur-sm hover:bg-white/20 transition-all duration-300 flex items-center justify-center gap-2"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clipRule="evenodd" />
                </svg>
                <span>View on GitHub</span>
                <div className="text-xs text-gray-400">18.2k ⭐</div>
              </a>
            </div>
          </div>

          {/* Right Column - Code Demo */}
          <div className="relative">
            {/* Glow Effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-orange-500/20 to-purple-500/20 rounded-3xl blur-2xl animate-pulse"></div>

            {/* Code Window */}
            <div className="relative transform hover:scale-105 transition-transform duration-700">
              <CodeWindow
                code={heroCode}
                language="python"
                fileName="production_ai_workflow.py"
              />
            </div>

            {/* Floating Elements */}
            <div className="absolute -top-8 -right-8 w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full blur-lg animate-bounce"></div>
            <div className="absolute -bottom-8 -left-8 w-12 h-12 bg-gradient-to-r from-orange-500 to-red-500 rounded-full blur-lg animate-bounce delay-1000"></div>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center">
          <div className="w-1 h-3 bg-white/50 rounded-full mt-2 animate-pulse"></div>
        </div>
      </div>
    </section>
  );
};

export default NewHero;
