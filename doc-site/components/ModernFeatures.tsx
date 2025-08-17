import React from 'react';

const ModernFeatures: React.FC = () => {
  const features = [
    {
      icon: '🚀',
      title: 'Production-Ready by Default',
      description: 'Built-in reliability patterns including circuit breakers, retries, timeouts, and graceful degradation.',
      benefits: ['99.9% uptime SLA', 'Auto-scaling', 'Zero-downtime deployments']
    },
    {
      icon: '🔍',
      title: 'Full Observability',
      description: 'Complete visibility into your AI workflows with distributed tracing, metrics, and real-time monitoring.',
      benefits: ['Distributed tracing', 'Real-time metrics', 'Custom dashboards']
    },
    {
      icon: '⚡',
      title: 'High-Performance Runtime',
      description: 'Optimized for concurrent AI workloads with intelligent batching and resource management.',
      benefits: ['50ms P95 latency', '10k+ RPS', 'Intelligent batching']
    },
    {
      icon: '🛡️',
      title: 'Enterprise Security',
      description: 'Built-in security features including authentication, authorization, and audit logging.',
      benefits: ['RBAC support', 'Audit logging', 'Data encryption']
    },
    {
      icon: '🔄',
      title: 'State Management',
      description: 'Persistent state management with automatic checkpointing and recovery from failures.',
      benefits: ['Auto-checkpointing', 'State recovery', 'Workflow versioning']
    },
    {
      icon: '📊',
      title: 'Cost Optimization',
      description: 'Intelligent resource allocation and cost monitoring to optimize your AI operations budget.',
      benefits: ['Resource optimization', 'Cost tracking', 'Usage analytics']
    }
  ];

  return (
    <section className="py-24 bg-gradient-to-b from-slate-900 to-slate-800 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-purple-600/5 via-pink-600/5 to-orange-600/5"></div>
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.02'%3E%3Cpath d='M20 20c0 4.4-3.6 8-8 8s-8-3.6-8-8 3.6-8 8-8 8 3.6 8 8zm0-20c0 4.4-3.6 8-8 8s-8-3.6-8-8 3.6-8 8-8 8 3.6 8 8zm20 0c0 4.4-3.6 8-8 8s-8-3.6-8-8 3.6-8 8-8 8 3.6 8 8zm0 20c0 4.4-3.6 8-8 8s-8-3.6-8-8 3.6-8 8-8 8 3.6 8 8z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
        }}></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Header */}
        <div className="text-center mb-20">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-full text-purple-300 text-sm font-medium backdrop-blur-sm mb-6">
            <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
            Enterprise Features
          </div>

          <h2 className="text-4xl lg:text-6xl font-black text-white mb-6 leading-tight">
            Everything You Need for
            <span className="bg-gradient-to-r from-orange-400 via-pink-500 to-purple-500 bg-clip-text text-transparent"> Production AI</span>
          </h2>

          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Stop cobbling together solutions. Get enterprise-grade AI infrastructure that scales with your business.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group relative bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 hover:border-white/20 transition-all duration-300 hover:transform hover:scale-105"
            >
              {/* Gradient background on hover */}
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

              <div className="relative z-10">
                {/* Icon */}
                <div className="text-5xl mb-6 group-hover:scale-110 transition-transform duration-300">
                  {feature.icon}
                </div>

                {/* Title */}
                <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-transparent group-hover:bg-gradient-to-r group-hover:from-orange-400 group-hover:to-pink-500 group-hover:bg-clip-text transition-all duration-300">
                  {feature.title}
                </h3>

                {/* Description */}
                <p className="text-gray-300 mb-6 leading-relaxed">
                  {feature.description}
                </p>

                {/* Benefits */}
                <div className="space-y-2">
                  {feature.benefits.map((benefit, benefitIndex) => (
                    <div key={benefitIndex} className="flex items-center gap-2 text-sm text-gray-400">
                      <div className="w-1.5 h-1.5 bg-gradient-to-r from-orange-400 to-pink-500 rounded-full"></div>
                      {benefit}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="text-center">
          <div className="inline-flex items-center gap-4 bg-gradient-to-r from-orange-500/10 to-pink-500/10 backdrop-blur-sm rounded-2xl p-8 border border-orange-500/20">
            <div className="text-4xl">🎯</div>
            <div className="text-left">
              <h4 className="text-xl font-bold text-white mb-2">Ready to see it in action?</h4>
              <p className="text-gray-300 mb-4">Deploy your first production AI workflow in under 5 minutes.</p>
              <a
                href="#quickstart"
                className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-orange-500 to-pink-500 text-white font-semibold rounded-xl hover:from-orange-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105"
              >
                Get Started Now
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModernFeatures;
