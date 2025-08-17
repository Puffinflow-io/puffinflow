import React from 'react';
import CodeBlock from './CodeBlock';

const quickStartCode = `# Install Puffinflow
pip install puffinflow

# Create your first production AI workflow
from puffinflow import Agent

agent = Agent("customer-support")

@agent.state
async def process_request(context):
    user_input = context.get_variable("user_input")

    # Your AI logic here
    response = await ai_model.process(user_input)

    context.set_variable("response", response)
    return "send_response"

@agent.state(checkpoint=True)
async def send_response(context):
    response = context.get_variable("response")
    await send_to_user(response)
    return None

# Run it
await agent.run(initial_context={"user_input": "Hello!"})`;

const NewQuickStart: React.FC = () => {
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, hash: string) => {
    e.preventDefault();
    window.location.hash = hash;
  };

  return (
    <section id="quickstart" className="quickstart-section">
      <div className="quickstart-container">
        <div className="quickstart-content">
          <div className="quickstart-header">
            <h2 className="quickstart-title">
              Get Started in Minutes
            </h2>
            <p className="quickstart-description">
              Transform your prototype into a production-ready AI workflow with just a few decorators.
            </p>
          </div>

          <div className="quickstart-grid">
            <div className="quickstart-steps">
              <div className="step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h3 className="step-title">Install Puffinflow</h3>
                  <p className="step-description">
                    Add Puffinflow to your existing Python project
                  </p>
                  <div className="step-code">
                    <code>pip install puffinflow</code>
                  </div>
                </div>
              </div>

              <div className="step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h3 className="step-title">Add Decorators</h3>
                  <p className="step-description">
                    Wrap your functions with @agent.state decorators
                  </p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h3 className="step-title">Deploy to Production</h3>
                  <p className="step-description">
                    Your workflow is now bulletproof and ready for scale
                  </p>
                </div>
              </div>
            </div>

            <div className="quickstart-code">
              <div className="code-window">
                <div className="code-window-header">
                  <div className="code-window-dots">
                    <span className="dot dot-red"></span>
                    <span className="dot dot-yellow"></span>
                    <span className="dot dot-green"></span>
                  </div>
                  <span className="code-window-title">quickstart.py</span>
                </div>
                <div className="code-window-body">
                  <CodeBlock code={quickStartCode} language="python" />
                </div>
              </div>
            </div>
          </div>

          <div className="quickstart-actions">
            <a
              href="#docs"
              onClick={(e) => handleNavClick(e, '#docs')}
              className="btn btn-primary"
            >
              View Documentation
            </a>
            <a
              href="https://github.com/puffinflow-io/puffinflow"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
            >
              Explore Examples
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default NewQuickStart;
