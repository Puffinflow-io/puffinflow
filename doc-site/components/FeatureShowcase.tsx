import React from 'react';

const FeatureShowcase: React.FC = () => {
  const features = [
    {
      icon: "🔧",
      title: "Zero-Config Setup",
      description: "Add one decorator and your code is production-ready. No architecture changes needed.",
      demo: {
        before: "@simple_function\ndef process_data():\n    return ai_model.predict()",
        after: "@agent.state(retry=3)\ndef process_data():\n    return ai_model.predict()"
      }
    },
    {
      icon: "📊",
      title: "Built-in Observability",
      description: "Track every state transition, error, and performance metric out of the box.",
      demo: {
        metrics: ["Response Time: 45ms", "Success Rate: 99.7%", "Error Rate: 0.3%", "Throughput: 1.2k/min"]
      }
    },
    {
      icon: "🔄",
      title: "Smart Auto-Recovery",
      description: "Intelligent retry logic, circuit breakers, and graceful degradation built in.",
      demo: {
        states: ["Processing", "Retry (1/3)", "Success", "Completed"]
      }
    }
  ];

  return (
    <section className="feature-showcase">
      <div className="showcase-container">
        <div className="showcase-header">
          <h2 className="showcase-title">
            Production AI Without the Production Headaches
          </h2>
          <p className="showcase-subtitle">
            Transform your AI prototype into enterprise-grade infrastructure in minutes, not months.
          </p>
        </div>

        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-showcase-card">
              <div className="feature-header">
                <div className="feature-icon-wrapper">
                  <span className="feature-icon">{feature.icon}</span>
                </div>
                <div className="feature-info">
                  <h3 className="feature-title">{feature.title}</h3>
                  <p className="feature-description">{feature.description}</p>
                </div>
              </div>

              <div className="feature-demo">
                {feature.demo.before && (
                  <div className="code-comparison">
                    <div className="code-block before">
                      <div className="code-header">Before</div>
                      <pre><code>{feature.demo.before}</code></pre>
                    </div>
                    <div className="code-arrow">→</div>
                    <div className="code-block after">
                      <div className="code-header">After</div>
                      <pre><code>{feature.demo.after}</code></pre>
                    </div>
                  </div>
                )}

                {feature.demo.metrics && (
                  <div className="metrics-display">
                    {feature.demo.metrics.map((metric, idx) => (
                      <div key={idx} className="metric-item">
                        <div className="metric-indicator"></div>
                        <span>{metric}</span>
                      </div>
                    ))}
                  </div>
                )}

                {feature.demo.states && (
                  <div className="state-flow">
                    {feature.demo.states.map((state, idx) => (
                      <div key={idx} className="state-item">
                        <div className="state-number">{idx + 1}</div>
                        <span className="state-label">{state}</span>
                        {idx < feature.demo.states!.length - 1 && <div className="state-arrow">→</div>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeatureShowcase;
