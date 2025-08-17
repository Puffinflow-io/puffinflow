import React from 'react';

const ModernStats: React.FC = () => {
  const stats = [
    {
      number: '99.9%',
      label: 'Uptime SLA',
      description: 'Enterprise reliability',
      icon: '⚡'
    },
    {
      number: '10k+',
      label: 'Requests/sec',
      description: 'High-performance processing',
      icon: '🚀'
    },
    {
      number: '50ms',
      label: 'P95 Latency',
      description: 'Lightning-fast responses',
      icon: '⚡'
    },
    {
      number: '500+',
      label: 'Companies',
      description: 'Trust Puffinflow',
      icon: '🏢'
    }
  ];

  const companies = [
    'TechCorp', 'DataFlow', 'AI Systems', 'CloudTech', 'NeuralNet', 'AutoScale'
  ];

  return (
    <section className="py-20 bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-full">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-r from-pink-500/10 to-orange-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        </div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="text-center group"
            >
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 hover:border-white/20 transition-all duration-300 hover:transform hover:scale-105">
                <div className="text-3xl mb-4 group-hover:scale-110 transition-transform duration-300">
                  {stat.icon}
                </div>
                <div className="text-4xl lg:text-5xl font-black text-white mb-2 group-hover:text-transparent group-hover:bg-gradient-to-r group-hover:from-orange-400 group-hover:to-pink-500 group-hover:bg-clip-text transition-all duration-300">
                  {stat.number}
                </div>
                <div className="text-lg font-semibold text-gray-300 mb-1">
                  {stat.label}
                </div>
                <div className="text-sm text-gray-400">
                  {stat.description}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Social Proof */}
        <div className="text-center mb-12">
          <p className="text-gray-400 mb-8 text-lg">
            Trusted by innovative companies worldwide
          </p>

          <div className="flex flex-wrap justify-center items-center gap-8 lg:gap-12">
            {companies.map((company, index) => (
              <div
                key={index}
                className="bg-white/5 backdrop-blur-sm rounded-xl px-6 py-3 border border-white/10 hover:border-white/20 transition-all duration-300 hover:transform hover:scale-105"
              >
                <div className="text-gray-300 font-semibold text-lg hover:text-white transition-colors duration-300">
                  {company}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Testimonial */}
        <div className="max-w-4xl mx-auto">
          <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 backdrop-blur-sm rounded-2xl p-8 lg:p-12 border border-purple-500/20">
            <div className="text-center">
              <div className="text-6xl mb-6">💬</div>
              <blockquote className="text-xl lg:text-2xl text-gray-300 mb-8 leading-relaxed">
                "Puffinflow transformed our AI operations from prototype to production in weeks, not months.
                The reliability and observability features are exactly what we needed for enterprise scale."
              </blockquote>
              <div className="flex items-center justify-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-r from-orange-400 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                  JS
                </div>
                <div className="text-left">
                  <div className="text-white font-semibold">Jane Smith</div>
                  <div className="text-gray-400 text-sm">Head of AI Engineering, TechCorp</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModernStats;
